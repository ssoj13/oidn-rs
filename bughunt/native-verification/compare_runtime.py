import hashlib, json, math, pathlib, struct, subprocess, sys, time
ROOT = pathlib.Path(__file__).parent
RUN = ROOT / 'matrix'
RUN.mkdir(exist_ok=True)
NATIVE = ROOT / 'runtime/oidn-2.5.0.x64.windows/bin/oidnDenoise.exe'
RUST = pathlib.Path('C:/projects/projects.rust.cg/cglibs/squarebob-rs/target/debug/oidn-rs.exe')
WEIGHTS = ROOT / 'oidn-weights'
def write_pfm(path, width, height, pixels):
    rows = [pixels[y * width * 3:(y + 1) * width * 3] for y in reversed(range(height))]
    flat = [v for row in rows for v in row]
    path.write_bytes(f'PF\n{width} {height}\n-1.0\n'.encode() + struct.pack('<' + 'f' * len(flat), *flat))
def read_pfm(path):
    with path.open('rb') as f:
        assert f.readline().strip() == b'PF'
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        assert scale == -1.0
        payload = f.read()
    assert len(payload) == w * h * 12
    values = struct.unpack('<' + 'f' * (w * h * 3), payload)
    return w, h, [v for y in reversed(range(h)) for v in values[y * w * 3:(y + 1) * w * 3]]
def stats(values):
    finite = [v for v in values if math.isfinite(v)]
    return dict(count=len(values), finiteCount=len(finite), nonfiniteCount=len(values)-len(finite), min=min(finite) if finite else None, max=max(finite) if finite else None, mean=sum(finite)/len(finite) if finite else None)
def metrics(actual, reference, input_values):
    masks = dict(all=[True] * (len(actual)//3), highlights=[max(input_values[i:i+3]) >= 5 for i in range(0,len(actual),3)], dark=[max(input_values[i:i+3]) < 1 for i in range(0,len(actual),3)])
    result = {}
    for name, mask in masks.items():
        pairs = [(a,b) for i,(a,b) in enumerate(zip(actual,reference)) if mask[i//3] and math.isfinite(a) and math.isfinite(b)]
        absolute = [abs(a-b) for a,b in pairs]
        relative = [abs(a-b)/max(abs(b),1e-6) for a,b in pairs]
        result[name] = dict(samples=len(pairs), maxAbsolute=max(absolute) if pairs else None, meanAbsolute=sum(absolute)/len(pairs) if pairs else None, maxRelative=max(relative) if pairs else None, meanRelative=sum(relative)/len(pairs) if pairs else None)
    return result
def fixture(kind,w,h):
    color, albedo, normal = [],[],[]
    for y in range(h):
        for x in range(w):
            if kind in ('uniform','tiny'): value=10.0
            elif kind == 'ramp': value=0.02+50*x/max(w-1,1)+0.2*y/max(h-1,1)
            elif kind == 'odd': value=20.0 if x >= w-3 or y >= h-2 else .05+.02*((x+y)%2)
            else:
                bright = w//4 <= x < 3*w//4 and h//4 <= y < 3*h//4
                value=(12+18*((x*13+y*17)%5)/4) if bright else .08+.04*((x*19+y*7)%3)/2
            color.extend((value,value*.7,value*.3))
            albedo.extend((.2+.5*x/max(w-1,1),.3+.2*y/max(h-1,1),.4))
            nx=((x%5)-2)*.15
            ny=((y%3)-1)*.2
            normal.extend((nx,ny,math.sqrt(1-nx*nx-ny*ny)))
    return color,albedo,normal
runs, comparisons = [],[]
cases=[('uniform',32,32,False),('ramp',32,32,False),('fine',64,48,False),('odd',33,19,False),('tiny',8,8,False),('fine_aov',33,19,True)]
alignment = '--alignment' in sys.argv
if alignment:
    cases=[('fine_aov_aligned',32,32,True),('fine_aov_oneaxis',32,19,True)]
result_path=ROOT/('alignment_comparison.json' if alignment else 'comparison.json')
started=time.time()
for kind,w,h,aov in cases:
    color,alb,nrm=fixture(kind,w,h)
    input_path=RUN/f'{kind}.pfm'
    write_pfm(input_path,w,h,color)
    alb_path=RUN/f'{kind}_albedo.pfm'
    nrm_path=RUN/f'{kind}_normal.pfm'
    if aov:
        write_pfm(alb_path,w,h,alb)
        write_pfm(nrm_path,w,h,nrm)
    weight=WEIGHTS/('rt_hdr_alb_nrm.tza' if aov else 'rt_hdr.tza')
    for scale in ([1,.02,None] if kind in ('tiny','odd') else [1,.02]):
        outputs={}
        scale_label='auto' if scale is None else str(scale)
        for engine,device,quality in [('rust','wgpu','balanced'),('native','cpu','balanced'),('native','cuda','balanced'),('native','cpu','high'),('native','cuda','high')]:
            label=f'{kind}_scale{scale_label}_{engine}_{device}_{quality}'
            output=RUN/f'{label}.pfm'
            if engine=='rust':
                cmd=[str(RUST),'denoise','--input',str(input_path),'--output',str(output),'--hdr','--quality',quality,'--weights',str(weight),'--verbose','0']
                if scale is not None: cmd+=['--input_scale',str(scale)]
                if aov: cmd+=['--albedo',str(alb_path),'--normal',str(nrm_path)]
            else:
                cmd=[str(NATIVE),'-d',device,'--hdr',str(input_path),'-o',str(output),'-q',quality,'-w',str(weight),'-v','1']
                if scale is not None: cmd+=['--is',str(scale)]
                if aov: cmd+=['--alb',str(alb_path),'--nrm',str(nrm_path)]
            begin=time.time()
            try:
                p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
                (RUN/f'{label}.stdout.log').write_text(p.stdout,encoding='utf-8')
                (RUN/f'{label}.stderr.log').write_text(p.stderr,encoding='utf-8')
                row=dict(case=kind,width=w,height=h,aov=aov,inputScale=scale,engine=engine,device=device,quality=quality,exitCode=p.returncode,seconds=time.time()-begin,command=cmd,output=str(output))
                if p.returncode==0:
                    ow,oh,data=read_pfm(output)
                    assert (ow,oh)==(w,h)
                    row.update(stats(data),sha256=hashlib.sha256(output.read_bytes()).hexdigest())
                    outputs[(engine,device,quality)]=data
                else: row['errorTail']=p.stderr[-1200:]
            except Exception as e:
                row=dict(case=kind,inputScale=scale,engine=engine,device=device,quality=quality,error=str(e),command=cmd)
            runs.append(row)
        pairs=[(('rust','wgpu','balanced'),('native','cpu','balanced')),(('rust','wgpu','balanced'),('native','cuda','balanced')),(('native','cuda','balanced'),('native','cpu','balanced')),(('native','cpu','high'),('native','cpu','balanced')),(('native','cuda','high'),('native','cuda','balanced'))]
        for actual,reference in pairs:
            if actual in outputs and reference in outputs:
                comparisons.append(dict(case=kind,width=w,height=h,aov=aov,inputScale=scale,actual=actual,reference=reference,metrics=metrics(outputs[actual],outputs[reference],color)))
        document=dict(format='PFM RGB f32 little-endian unit magnitude LF header; top-down comparisons',fixturePolicy='deterministic finite HDR and optional raw linear albedo/signed normals',confounders=['native high can change execution precision despite forced identical Base weights','CLI mutable host adapter, not Squarebob immutable bridge'],elapsedSeconds=time.time()-started,runs=runs,comparisons=comparisons)
        result_path.write_text(json.dumps(document,indent=2),encoding='utf-8')
        print(f'{kind} scale={scale_label} completed {len(runs)} runs',flush=True)
print(json.dumps(dict(runs=len(runs),failures=sum(r.get('exitCode',1)!=0 for r in runs),comparisons=len(comparisons),seconds=time.time()-started)))
