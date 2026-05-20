from __future__ import annotations

"""Compress/decompress git repository folders for offline dataset extraction."""

import zipfile
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Compress/decompress repository folders.")


def validate(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise typer.BadParameter(f"The root path does not exist: {path}")
    if not path.is_dir():
        raise typer.BadParameter(f"The root path is not a directory: {path}")
    return path


@app.command(name="compress")
def compress(root: Path = typer.Option(..., "--root", callback=lambda value: validate(str(value)), help="root directory")):
    for item in root.iterdir():
        if item.is_dir():
            filename = f"{item.name}.zip"
            path = root / filename
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in item.rglob("*"):
                    if file.is_file():
                        zf.write(file, file.relative_to(item))
            typer.echo(f"Finished {filename}")
    typer.echo("All folders have been compressed")


@app.command(name="decompress")
def decompress(root: Path = typer.Option(..., "--root", callback=lambda value: validate(str(value)), help="root directory")):
    zip_files = list(root.glob("*.zip"))
    if not zip_files:
        typer.echo("No zip files found in root directory.")
        return
    for zip_path in zip_files:
        folder_name = zip_path.stem
        extract_dir = root / folder_name
        if extract_dir.exists():
            typer.echo(f"Folder already exists: {extract_dir}. Skipping ...")
            continue
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        typer.echo(f"Decompressing {zip_path.name} -> {extract_dir}")
    typer.echo("All zip files have been decompressed")


__all__ = ["app", "compress", "decompress", "validate"]
