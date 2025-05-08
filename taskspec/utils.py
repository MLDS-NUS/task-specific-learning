import os
from pathlib import Path
def is_jupyter_notebook():
    return 'ipykernel' in os.sys.modules

def find_project_root(markers=('pyproject.toml', )) -> Path:
    try:
        current = Path(__file__).resolve()
    except NameError:
        current = Path().resolve()
    
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return str(parent)
    raise FileNotFoundError(
        f"Not found"
    )
# 
