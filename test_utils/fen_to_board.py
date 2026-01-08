import requests
from urllib.parse import quote

def fen_to_board_image(
    fen: str,
    out_path: str,
    color: str = "white"
):
    """
    Download a chessboard image from Lichess given a FEN string.

    Parameters:
        fen (str): FEN string (board-only or full FEN)
        out_path (str): where to save the image (e.g. 'board.gif')
        color (str): 'white' or 'black' (board orientation)
    """

    # Lichess expects spaces replaced with underscores
    fen_for_url = fen.replace(" ", "_")

    url = (
        "https://lichess1.org/export/fen.gif"
        f"?fen={quote(fen_for_url, safe='/_')}"
        f"&color={color}"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(response.content)

    return out_path

