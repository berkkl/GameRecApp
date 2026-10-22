from src.gamerec.models import Game


def test_game_model():
    game = Game(id="1", title="Test", description="Desc", metadata={})
    assert game.title == "Test"
    assert game.tags == []


def test_api_import():
    from src.gamerec.serve.api import app

    assert app.title == "GameRecApp API"
