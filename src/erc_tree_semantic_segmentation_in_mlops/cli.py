"""Console script for erc_tree_semantic_segmentation_in_mlops."""
import erc_tree_semantic_segmentation_in_mlops

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for erc_tree_semantic_segmentation_in_mlops."""
    console.print("Replace this message by putting your code into "
               "erc_tree_semantic_segmentation_in_mlops.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
