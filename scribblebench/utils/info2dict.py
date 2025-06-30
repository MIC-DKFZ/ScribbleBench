
def info2dict(filepath):
    with open((filepath.parent / "Info.cfg"), "r") as f:
        info = f.read()

    data = {}
    for line in info.strip().splitlines():
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        # Convert value to int or float if possible
        if value.replace('.', '', 1).isdigit():
            value = float(value) if '.' in value else int(value)

        data[key] = value
    return data