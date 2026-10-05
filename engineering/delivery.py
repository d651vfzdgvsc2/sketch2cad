"""One final CAD file per engineering result; diagnostics stay internal."""
from pathlib import Path
import shutil
import ezdxf


def publish_result(selected, calibration, folder, stem):
    source=Path(calibration.get('dxf') or selected['dxf'])
    doc=ezdxf.readfile(source)
    if doc.audit().has_errors:
        raise ValueError('Cannot deliver a DXF with audit errors')
    # Never clean a user's folder or overwrite previous runs.
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    target=folder/f'{stem}.dxf'
    try:
        shutil.copy2(source,target)
    except Exception:
        if target.exists():target.unlink()
        folder.rmdir()
        raise
    return dict(directory=str(folder),dxf=str(target),units='mm' if calibration.get('dxf') else 'pixels')
