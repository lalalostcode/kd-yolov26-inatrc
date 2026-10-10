"""Buat sembilan notebook dari runner bersama; --check memeriksa tanpa menulis."""

import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path


# Path mengikuti lokasi script, sehingga perintah dapat dipanggil dari folder lain.
REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "notebooks" / "01_kaggle_runner.ipynb"
OUTPUT_DIR = REPO_ROOT / "notebooks" / "experiments"

# Ini hanya sembilan pilihan tetap; mekanisme training tetap di package inatrc_kd.
PRESETS = (
    {"name": "01_teacher_yolo26m", "title": "Teacher YOLO26m", "stage": "teacher", "student": "n", "method": "none"},
    {"name": "02_baseline_yolo26s", "title": "Baseline YOLO26s", "stage": "student", "student": "s", "method": "none"},
    {"name": "03_baseline_yolo26n", "title": "Baseline YOLO26n", "stage": "student", "student": "n", "method": "none"},
    {"name": "04_native_yolo26s", "title": "Native KD YOLO26s", "stage": "student", "student": "s", "method": "native"},
    {"name": "05_native_yolo26n", "title": "Native KD YOLO26n", "stage": "student", "student": "n", "method": "native"},
    {"name": "06_crosskd_yolo26s", "title": "CrossKD YOLO26s", "stage": "student", "student": "s", "method": "crosskd"},
    {"name": "07_crosskd_yolo26n", "title": "CrossKD YOLO26n", "stage": "student", "student": "n", "method": "crosskd"},
    {"name": "08_csakd_yolo26s", "title": "CSAKD YOLO26s", "stage": "student", "student": "s", "method": "csakd"},
    {"name": "09_csakd_yolo26n", "title": "CSAKD YOLO26n", "stage": "student", "student": "n", "method": "csakd"},
)


def _source(cell):
    """Notebook JSON boleh menyimpan source sebagai string atau daftar baris."""
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def _set_source(cell, source):
    cell["source"] = source.splitlines(keepends=True)


def render_notebook(template, preset):
    """Salin runner, kunci preset, dan bersihkan output tanpa mengubah template."""
    notebook = deepcopy(template)
    values = {"STAGE": preset["stage"], "STUDENT": preset["student"], "METHOD": preset["method"]}
    config_cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(_source(cell))
        assignments = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in values:
                    if name in assignments:
                        raise ValueError(f"Assignment {name} ganda pada template.")
                    assignments[name] = node
        if assignments:
            if set(assignments) != set(values):
                raise ValueError("STAGE, STUDENT, dan METHOD harus berada dalam satu cell konfigurasi.")
            config_cells.append((cell, assignments))

    if len(config_cells) != 1:
        raise ValueError("Template harus mempunyai tepat satu cell konfigurasi eksperimen.")

    config, assignments = config_cells[0]
    lines = _source(config).splitlines(keepends=True)
    comments = {
        "STAGE": "preset tetap; gunakan notebook lain untuk eksperimen berbeda",
        "STUDENT": "diabaikan saat teacher; teacher selalu YOLO26m" if preset["stage"] == "teacher" else "preset model notebook ini",
        "METHOD": "preset metode notebook ini",
    }
    for name, node in assignments.items():
        if node.lineno != node.end_lineno or not isinstance(node.value, ast.Constant):
            raise ValueError(f"Assignment {name} pada template harus literal satu baris.")
        lines[node.lineno - 1] = f'{name} = {json.dumps(values[name])}  # {comments[name]}\n'
    fixed = (preset["stage"], preset["student"], preset["method"])
    config_source = "".join(lines).rstrip() + "\n\n"
    config_source += "# Cegah eksperimen berubah tanpa sengaja; pilih file notebook yang sesuai.\n"
    config_source += f"assert (STAGE, STUDENT, METHOD) == {fixed!r}, (\n"
    message = f'Preset {preset["name"]} harus tetap: stage={preset["stage"]}, student={preset["student"]}, method={preset["method"]}. Pilih notebook lain untuk eksperimen berbeda.'
    config_source += f"    {json.dumps(message, ensure_ascii=False)}\n)\n"
    _set_source(config, config_source)

    if not notebook["cells"] or notebook["cells"][0]["cell_type"] != "markdown":
        raise ValueError("Template harus dimulai dengan markdown pengantar.")
    introduction = _source(notebook["cells"][0]).splitlines(keepends=True)
    if not introduction or not introduction[0].startswith("# "):
        raise ValueError("Pengantar template harus mempunyai judul markdown.")
    introduction[0] = f'# InaTRC · {preset["title"]}\n'
    intro = "".join(introduction)
    intro = intro.replace("Runner satu eksperimen.", f'Notebook khusus **{preset["title"]}**, satu eksperimen per Run All.', 1)
    intro = intro.replace(
        "Runner umum ini menjadi template sembilan notebook di `notebooks/experiments/`. Pilih notebook eksperimen untuk memakai preset model/metode yang tetap.",
        "Notebook ini dibuat dari runner umum; setup memakai template yang sama dan training tetap di repository.",
        1,
    )
    intro += "\n\nStage, model, dan metode sudah tetap. Ubah seed, mode, sumber repository, dataset, checkpoint teacher, device, dan W&B sesuai kebutuhan. Default **smoke, seed 42, W&B disabled**."
    _set_source(notebook["cells"][0], intro)

    # Semua notebook berbagi cell setup yang sama dan bebas hasil eksekusi sebelumnya.
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile(_source(cell), preset["name"], "exec")
            cell["execution_count"] = None
            cell["outputs"] = []
    model = "yolo26m" if preset["stage"] == "teacher" else "yolo26" + preset["student"]
    notebook.setdefault("metadata", {})["title"] = preset["title"]
    notebook["metadata"]["inatrc_experiment"] = {
        "preset": preset["name"],
        "stage": preset["stage"],
        "student": preset["student"],
        "model": model,
        "method": preset["method"],
    }
    return notebook


def _rendered_notebooks(template_path):
    template = json.loads(Path(template_path).read_text(encoding="utf-8"))
    for preset in PRESETS:
        rendered = render_notebook(template, preset)
        content = json.dumps(rendered, ensure_ascii=False, indent=2) + "\n"
        yield preset["name"] + ".ipynb", content


def generate_notebooks(template_path=TEMPLATE_PATH, output_dir=OUTPUT_DIR):
    """Tulis sembilan file deterministik; jangan mengubah notebook manual lain."""
    rendered = list(_rendered_notebooks(template_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, content in rendered:
        path = output_dir / filename
        path.write_text(content, encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def check_notebooks(template_path=TEMPLATE_PATH, output_dir=OUTPUT_DIR):
    """Kembalikan nama file missing/outdated; pemeriksaan tidak menulis apa pun."""
    outdated = []
    for filename, content in _rendered_notebooks(template_path):
        path = Path(output_dir) / filename
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            outdated.append(filename)
    return outdated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Periksa kesamaan hasil generator tanpa menulis file.")
    args = parser.parse_args(argv)
    if args.check:
        outdated = check_notebooks()
        if outdated:
            print("Notebook missing/outdated: " + ", ".join(outdated))
            print("Jalankan: python scripts/generate_experiment_notebooks.py")
            return 1
        print("Sembilan notebook sesuai template.")
        return 0
    paths = generate_notebooks()
    print(f"Dibuat {len(paths)} notebook di {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
