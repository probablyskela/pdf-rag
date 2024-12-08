import typing


def chunk_document(
    doc: str,
    chunk_size: int = 512,
    max_chunk_size: int = 600,
) -> typing.Generator[str, None, None]:
    chunk = ""
    for line in doc.splitlines():
        chunk += f"{line}\n"
        if len(chunk) >= chunk_size:
            yield chunk[:max_chunk_size]
            chunk = ""
    if chunk:
        yield chunk[:max_chunk_size]
