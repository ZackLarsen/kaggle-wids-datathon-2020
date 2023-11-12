from hydra_zen import builds

def foo(bar: int, baz: list[str], qux: float = 1.23):
    return None

ZenFooConf = builds(foo, bar=2, baz=["abc"], populate_full_signature=True)

print(ZenFooConf.bar)
