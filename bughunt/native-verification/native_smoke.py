import json, math, pathlib, struct, subprocess
root = pathlib.Path(__file__).parent
width = height = 32
values = []
for y in range(height):
    for x in range(width):
        bright = 20.0 if 12 <= x < 20 and 12 <= y < 20 else 0.1 + 0.05 * ((x + y) % 2)
        values.extend((bright, bright * 0.7, bright * 0.3))
input_path = root / 'smoke_input.pfm'
input_path.write_bytes(f'PF\n{width} {height}\n-1.0\n'.encode() + struct.pack('<' + 'f' * len(values), *values))
exe = root / 'runtime/oidn-2.5.0.x64.windows/bin/oidnDenoise.exe'
weights = root / 'oidn-weights/rt_hdr.tza'
results = []
for device in ('cpu', 'cuda'):
    output = root / f'smoke_{device}.pfm'
    command = [str(exe), '-d', device, '--hdr', str(input_path), '--is', '1', '-q', 'balanced', '-w', str(weights), '-o', str(output), '-v', '1']
    process = subprocess.run(command, capture_output=True, text=True)
    (root / f'smoke_{device}.stdout.log').write_text(process.stdout, encoding='utf-8')
    (root / f'smoke_{device}.stderr.log').write_text(process.stderr, encoding='utf-8')
    record = {'device': device, 'command': command, 'exitCode': process.returncode}
    if process.returncode == 0:
        data = output.read_bytes().split(b'\n', 3)[3]
        rgb = struct.unpack('<' + 'f' * (len(data) // 4), data)
        record.update(count=len(rgb), finite=all(math.isfinite(v) for v in rgb), min=min(rgb), max=max(rgb))
    results.append(record)
(root / 'native_smoke.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results))
