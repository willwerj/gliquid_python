from pathlib import Path
_DIR_STRUCT_OPTS = ['flat', 'nested']

project_root = None
data_dir = None
dir_structure = None
phase_transitions_file = None

def set_project_root(path: Path):
    global project_root
    project_root = path

def set_data_dir(path: Path):
    global data_dir
    global phase_transitions_file

    data_dir = path
    phase_transitions_file = Path(data_dir / "phase_transitions.json")

def set_dir_structure(structure: str):
    global dir_structure
    if structure not in _DIR_STRUCT_OPTS:
        raise ValueError(f"dir_structure must be one of {_DIR_STRUCT_OPTS}")
    dir_structure = structure
   
def find_project_root(dirname = 'gliquid_python'):
    current = Path.cwd()
    if current.name == dirname:
        return current
    for parent in current.parents:
        if parent.name == dirname:
            return parent
    return current

set_project_root(find_project_root())
set_data_dir(Path(project_root / "data"))
set_dir_structure(_DIR_STRUCT_OPTS[0])
