"""High-precision document parsing engine using IBM Docling."""

import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional, Union

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice as DlAcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument
from loguru import logger

from veritas_crag.config.settings import AcceleratorDevice, settings
from veritas_crag.core.exceptions import (
    DocumentNotFoundError,
    DocumentParsingError,
    UnsupportedDocumentFormatError,
)
from veritas_crag.ingestion.chunker import DoclingChunker
from veritas_crag.models.document import (
    DocumentMetadata,
    DocumentTable,
    ParsedDocument,
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".md", ".asciidoc"}


class DoclingParser:
    """Production-grade Document Parser using Docling with hardware acceleration support."""

    def __init__(
        self,
        enable_ocr: Optional[bool] = None,
        enable_table_structure: Optional[bool] = None,
        accelerator_device: Optional[AcceleratorDevice] = None,
        num_threads: Optional[int] = None,
        generate_page_images: Optional[bool] = None,
    ):
        self.enable_ocr = enable_ocr if enable_ocr is not None else settings.ENABLE_OCR
        self.enable_table_structure = (
            enable_table_structure
            if enable_table_structure is not None
            else settings.ENABLE_TABLE_STRUCTURE
        )
        self.device = accelerator_device or settings.ACCELERATOR_DEVICE
        self.num_threads = num_threads or settings.NUM_THREADS
        self.generate_page_images = (
            generate_page_images
            if generate_page_images is not None
            else settings.GENERATE_PAGE_IMAGES
        )

        self._converter = self._build_converter()
        self._chunker = DoclingChunker()

    def _map_accelerator_device(self) -> DlAcceleratorDevice:
        """Maps config AcceleratorDevice to Docling's AcceleratorDevice enum."""
        device_mapping = {
            AcceleratorDevice.AUTO: DlAcceleratorDevice.AUTO,
            AcceleratorDevice.MPS: DlAcceleratorDevice.MPS,
            AcceleratorDevice.CUDA: DlAcceleratorDevice.CUDA,
            AcceleratorDevice.CPU: DlAcceleratorDevice.CPU,
        }
        return device_mapping.get(self.device, DlAcceleratorDevice.AUTO)

    def _build_converter(self) -> DocumentConverter:
        """Initializes and configures the Docling DocumentConverter."""
        try:
            accelerator_opts = AcceleratorOptions(
                num_threads=self.num_threads,
                device=self._map_accelerator_device(),
            )

            pipeline_options = PdfPipelineOptions(
                do_ocr=self.enable_ocr,
                do_table_structure=self.enable_table_structure,
                accelerator_options=accelerator_opts,
                generate_page_images=self.generate_page_images,
            )

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info(
                f"Initialized DoclingParser (Device: {self.device.value}, OCR: {self.enable_ocr}, "
                f"Tables: {self.enable_table_structure})"
            )
            return converter

        except Exception as e:
            logger.error(f"Failed to initialize Docling DocumentConverter: {e}")
            raise DocumentParsingError(f"Converter initialization failed: {e}") from e

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        """Computes SHA-256 hash for document content tracking and cache validation."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def parse(
        self,
        file_path: Union[str, Path],
        chunk: bool = True,
    ) -> ParsedDocument:
        """Parses a document file and extracts structured content, tables, and chunks.

        Args:
            file_path: Path to the target document.
            chunk: Whether to automatically generate layout-aware chunks.

        Returns:
            ParsedDocument containing metadata, markdown, full text, tables, and chunks.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            logger.error(f"File not found: {path}")
            raise DocumentNotFoundError(f"Document file does not exist: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.error(f"Unsupported document format: {path.suffix}")
            raise UnsupportedDocumentFormatError(
                f"Unsupported format '{path.suffix}'. Supported formats: {SUPPORTED_EXTENSIONS}"
            )

        file_size = path.stat().st_size
        logger.info(f"Parsing document '{path.name}' ({file_size / (1024 * 1024):.2f} MB)...")
        start_time = time.perf_counter()

        try:
            conversion_res = self._converter.convert(path)
            doc: DoclingDocument = conversion_res.document
            parsing_duration = time.perf_counter() - start_time

            # Compute SHA-256
            file_hash = self._compute_file_hash(path)

            # Export Markdown & plain text
            markdown_content = doc.export_to_markdown()
            full_text = doc.export_to_text()

            # Extract tables
            extracted_tables: List[DocumentTable] = []
            for idx, tbl in enumerate(doc.tables):
                # Page numbers for table
                tbl_pages: List[int] = []
                if hasattr(tbl, "prov") and tbl.prov:
                    for prov in tbl.prov:
                        if hasattr(prov, "page_no") and prov.page_no not in tbl_pages:
                            tbl_pages.append(prov.page_no)

                # Export markdown / HTML / dataframe matrix
                tbl_md = tbl.export_to_markdown(doc=doc)
                tbl_html = tbl.export_to_html(doc=doc)

                df_matrix = None
                num_rows = 0
                num_cols = 0
                try:
                    df = tbl.export_to_dataframe(doc=doc)
                    num_rows, num_cols = df.shape
                    df_matrix = [df.columns.tolist()] + df.values.tolist()
                except Exception as ex:
                    logger.debug(f"Could not convert table {idx} to dataframe: {ex}")

                extracted_tables.append(
                    DocumentTable(
                        table_index=idx,
                        page_numbers=tbl_pages,
                        num_rows=num_rows,
                        num_cols=num_cols,
                        markdown=tbl_md,
                        html=tbl_html,
                        data=df_matrix,
                    )
                )

            # Extract page count
            num_pages = len(getattr(doc, "pages", [])) or 1

            # Build metadata
            metadata = DocumentMetadata(
                source_path=str(path),
                file_name=path.name,
                file_size_bytes=file_size,
                num_pages=num_pages,
                title=getattr(doc, "name", path.stem),
                content_hash=file_hash,
                parsing_duration_seconds=round(parsing_duration, 3),
            )

            # Generate chunks if requested
            chunks = []
            if chunk:
                chunks = self._chunker.chunk_document(
                    docling_doc=doc,
                    doc_source=str(path),
                    doc_hash=file_hash,
                )

            parsed_doc = ParsedDocument(
                metadata=metadata,
                full_text=full_text,
                markdown=markdown_content,
                tables=extracted_tables,
                chunks=chunks,
            )

            logger.info(
                f"Successfully parsed '{path.name}' in {parsing_duration:.2f}s "
                f"({len(extracted_tables)} tables, {len(chunks)} chunks, {num_pages} pages)"
            )
            return parsed_doc

        except Exception as e:
            logger.error(f"Failed to parse document '{path}': {e}")
            raise DocumentParsingError(f"Error parsing '{path.name}': {e}") from e

    def export_to_disk(
        self,
        parsed_doc: ParsedDocument,
        output_dir: Optional[Union[str, Path]] = None,
        export_markdown: bool = True,
        export_json: bool = True,
        export_chunks: bool = True,
    ) -> Path:
        """Exports parsed document artifacts to specified output directory."""
        target_dir = Path(output_dir or settings.OUTPUT_DIR).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(parsed_doc.metadata.file_name).stem

        # Save Markdown
        if export_markdown:
            md_path = target_dir / f"{stem}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(parsed_doc.markdown)
            logger.info(f"Exported Markdown to: {md_path}")

        # Save Full JSON
        if export_json:
            json_path = target_dir / f"{stem}_parsed.json"
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(parsed_doc.model_dump_json(indent=2))
            logger.info(f"Exported JSON to: {json_path}")

        # Save Chunks JSON
        if export_chunks and parsed_doc.chunks:
            chunks_path = target_dir / f"{stem}_chunks.json"
            with open(chunks_path, "w", encoding="utf-8") as f:
                chunks_data = [chunk.model_dump() for chunk in parsed_doc.chunks]
                json.dump(chunks_data, f, indent=2)
            logger.info(f"Exported Chunks to: {chunks_path}")

        return target_dir
