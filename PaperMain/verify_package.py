"""Verify frozen artifact copies and, if available, the final package manifest."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=json.loads((ROOT/'provenance/source_manifest.json').read_text())
    errors=[]
    for e in source['entries']:
        p=ROOT/e['path']
        if not p.is_file() or p.is_symlink() or sha(p)!=e['sha256']: errors.append(e['path'])
    count=0
    if (ROOT/'PACKAGE_MANIFEST.json').exists():
        manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
        for name,want in manifest['sha256'].items():
            count+=1;p=ROOT/name
            if not p.is_file() or p.is_symlink() or sha(p)!=want: errors.append(name)
    print(json.dumps({'original_artifacts_verified':len(source['entries']),
                      'package_files_verified':count,'errors':sorted(set(errors))},indent=2))
    if errors: raise SystemExit(1)
if __name__=='__main__': main()
