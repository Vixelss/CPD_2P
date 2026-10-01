"""Prueba minima: el paquete se importa."""


def test_importa_paquete() -> None:
    import pdn

    assert pdn is not None
