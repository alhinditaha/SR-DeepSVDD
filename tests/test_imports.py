def test_imports():
    # Smoke import test for package discovery
    try:
        import src
    except Exception as e:
        raise AssertionError(f'Package import failed: {e}')
