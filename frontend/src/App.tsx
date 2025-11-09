import React, { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";

/**
 * ECE 30861 / 46100 — Frontend Starter (React + TypeScript + Tailwind)
 * Wired to: ece461_fall_2025_openapi_spec.yaml (OpenAPI 3.0.2)
 *
 * Features in this single-file demo:
 * - Login via PUT /authenticate → gets token (string) to send as X-Authorization header.
 * - Artifact search via POST /artifacts (supports name / "*" and types[] filters).
 * - View artifact by id via GET /artifacts/{artifact_type}/{id}.
 * - Create artifact via POST /artifact/{artifact_type} with { url }.
 * - Delete artifact via DELETE /artifact/{artifact_type}/{id} (NON-BASELINE, optional).
 * - Model endpoints: GET /artifact/model/{id}/rate, POST /artifact/model/{id}/license-check, GET /artifact/model/{id}/lineage.
 * - Search helpers: GET /artifact/byName/{name}, POST /artifact/byRegEx.
 * - Basic ADA/WCAG care: labeled inputs, aria-live for errors, skip link, focus-visible rings.
 *
 * HOW TO USE
 * 1) Create a Vite React + TS app (or drop into existing):
 *    npm create vite@latest model-registry-ui -- --template react-ts
 *    cd model-registry-ui && npm i && npm i -D tailwindcss postcss autoprefixer
 *    npx tailwindcss init -p
 *    - Configure Tailwind content in tailwind.config.js to scan src/**/*.tsx
 *    - Add Tailwind directives to src/index.css: @tailwind base; @tailwind components; @tailwind utilities;
 * 2) Replace src/App.tsx contents with this file, or create src/App.tsx and export default App.
 * 3) Define API base in .env: VITE_API_BASE=https://YOUR_BACKEND_HOST
 * 4) npm run dev
 */

// ===================== Config =====================
const API_BASE = (import.meta as any)?.env?.VITE_API_BASE?.replace(/\/$/, "") || "http://localhost:8080";

// ===================== Types (subset from spec) =====================
export type ArtifactType = "model" | "dataset" | "code";
export interface ArtifactMetadata { name: string; id: string; type: ArtifactType }
export interface Artifact { metadata: ArtifactMetadata; data: { url: string } }
export interface ArtifactQuery { name: string; types?: ArtifactType[] }
export interface ModelRating {
  name: string; category: string; net_score: number; net_score_latency: number;
  ramp_up_time: number; ramp_up_time_latency: number;
  bus_factor: number; bus_factor_latency: number;
  performance_claims: number; performance_claims_latency: number;
  license: number; license_latency: number;
  dataset_and_code_score: number; dataset_and_code_score_latency: number;
  dataset_quality: number; dataset_quality_latency: number;
  code_quality: number; code_quality_latency: number;
  reproducibility: number; reproducibility_latency: number;
  reviewedness: number; reviewedness_latency: number;
  tree_score: number; tree_score_latency: number;
  size_score: { raspberry_pi: number; jetson_nano: number; desktop_pc: number; aws_server: number };
  size_score_latency: number;
}
export interface ArtifactLineageGraph {
  nodes: { artifact_id: string; name: string; source: string; metadata?: Record<string, any> }[];
  edges: { from_node_artifact_id: string; to_node_artifact_id: string; relationship: string }[];
}

// ===================== API helper =====================
async function api<T>(path: string, opts: {
  method?: string; body?: any; token?: string; headers?: Record<string,string>;
} = {}): Promise<T> {
  const { method = "GET", body, token, headers = {} } = opts;
  const url = `${API_BASE}${path}`;
  const h: Record<string,string> = { Accept: "application/json", ...headers };
  const init: RequestInit = { method, headers: h };
  if (token) h["X-Authorization"] = token; // per spec: all endpoints use X-Authorization
  if (body !== undefined) {
    h["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(url, init);
  const ct = res.headers.get("content-type") || "";
  const parse = async () => ct.includes("application/json") ? res.json() : res.text();
  if (!res.ok) {
    const err = await parse();
    const message = typeof err === "string" ? err : err?.message || `HTTP ${res.status}`;
    throw new Error(message);
  }
  return (await parse()) as T;
}

// ===================== UI widgets =====================
function FieldLabel({ id, children }: { id: string; children: React.ReactNode }) {
  return <label htmlFor={id} className="block text-sm font-medium text-gray-800">{children}</label>;
}
function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={"mt-1 w-full rounded-xl border px-3 py-2 outline-none focus-visible:ring-2 focus-visible:ring-blue-600 " + (props.className||"")} />
}
function Button(props: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button {...props} className={"rounded-2xl px-4 py-2 shadow-sm focus-visible:ring-2 focus-visible:ring-blue-600 disabled:opacity-50 " + (props.className||"bg-blue-600 text-white hover:bg-blue-700")}/>;
}
function Card({ title, children, actions }: { title?: string; children: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <section className="rounded-2xl border bg-white p-5 shadow-sm">
      {title && <h2 className="mb-3 text-lg font-semibold text-gray-900">{title}</h2>}
      <div>{children}</div>
      {actions && <div className="mt-4 flex gap-2">{actions}</div>}
    </section>
  );
}

// Accessible skip link
function SkipToContent(){
  return <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:rounded-md focus:bg-yellow-200 focus:px-3 focus:py-1">Skip to main content</a>;
}

// ===================== App =====================
export default function App(){
  const [token, setToken] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<"login"|"search"|"detail"|"create"|"rate"|"lineage"|"license"|"regex"|"byname">("login");
  const [selected, setSelected] = useState<ArtifactMetadata | null>(null);

  useEffect(()=>{ setError(null); }, [view]);

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <SkipToContent/>
      <header className="sticky top-0 z-10 border-b bg-white/90 backdrop-blur">
        <nav className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <span aria-hidden className="inline-block h-7 w-7 rounded-xl bg-blue-600"/>
            <h1 className="text-xl font-semibold">ACME Model Registry</h1>
          </div>
          <div className="flex items-center gap-2">
            {token && (
              <>
                <Button onClick={()=>setView("search")} className="bg-white text-gray-900 border hover:bg-gray-100">Search</Button>
                <Button onClick={()=>setView("create")} className="bg-white text-gray-900 border hover:bg-gray-100">Create</Button>
                {selected && <Button onClick={()=>setView("detail")} className="bg-white text-gray-900 border hover:bg-gray-100">Detail</Button>}
                <Button onClick={()=>{setToken(null); setSelected(null); setView("login");}} className="bg-red-600 hover:bg-red-700">Logout</Button>
              </>
            )}
          </div>
        </nav>
      </header>

      <main id="main" className="mx-auto grid max-w-6xl gap-6 px-4 py-6">
        {error && <div role="alert" aria-live="assertive" className="rounded-xl border border-red-300 bg-red-50 p-3 text-red-900">{error}</div>}
        {!token ? <Login onAuthed={setToken} onError={setError}/> : (
          {
            login: null,
            search: <Search token={token} onError={setError} onPick={(m)=>{ setSelected(m); setView("detail"); }}/>,
            detail: <Detail token={token} meta={selected} onError={setError} onOpen={(tab)=>setView(tab)}/>,
            create: <Create token={token} onError={setError} onCreated={(m)=>{ setSelected(m.metadata); setView("detail"); }}/>,
            rate: selected ? <Rate token={token} id={selected.id}/> : null,
            lineage: selected ? <Lineage token={token} id={selected.id}/> : null,
            license: selected ? <LicenseCheck token={token} id={selected.id}/> : null,
            regex: <RegexSearch token={token} onPick={(m)=>{ setSelected(m); setView("detail"); }}/>,
            byname: <ByName token={token} onPick={(m)=>{ setSelected(m); setView("detail"); }}/>,
          }[view]
        )}
      </main>

      <footer className="mx-auto max-w-6xl px-4 py-6 text-sm text-gray-600">ECE 30861 · WCAG 2.1 AA-minded UI · Using X-Authorization header</footer>
    </div>
  );
}

// ===================== Screens =====================
function Login({ onAuthed, onError }: { onAuthed: (t:string)=>void; onError:(m:string)=>void }){
  const [username, setUsername] = useState("ece30861defaultadminuser");
  const [password, setPassword] = useState("correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;");
  const [busy, setBusy] = useState(false);
  async function submit(e: FormEvent){
    e.preventDefault();
    setBusy(true);
    try{
      // PUT /authenticate { user:{name,is_admin?}, secret:{password} }
      const token = await api<string>("/authenticate", {
        method: "PUT",
        body: { user: { name: username, is_admin: true }, secret: { password } },
      });
      // The example shows a quoted string; strip surrounding quotes if present
      const cleaned = (token as any).replace?.(/^"|"$/g, "") ?? token;
      onAuthed(cleaned);
    }catch(err:any){ onError(err.message || String(err)); }
    finally{ setBusy(false); }
  }
  return (
    <Card title="Sign in">
      <form onSubmit={submit} className="grid gap-3" aria-describedby="login-help">
        <p id="login-help" className="text-sm text-gray-600">Enter the course default credentials or a user provisioned by your backend. Token will be used as <code>X-Authorization</code>.</p>
        <div>
          <FieldLabel id="u">Username</FieldLabel>
          <Input id="u" autoComplete="username" value={username} onChange={e=>setUsername(e.target.value)} required/>
        </div>
        <div>
          <FieldLabel id="p">Password</FieldLabel>
          <Input id="p" type="password" autoComplete="current-password" value={password} onChange={e=>setPassword(e.target.value)} required/>
        </div>
        <div className="flex gap-2">
          <Button disabled={busy} type="submit">{busy ? "Signing in…" : "Sign in"}</Button>
        </div>
      </form>
    </Card>
  );
}

function Search({ token, onError, onPick }:{ token:string; onError:(m:string)=>void; onPick:(m:ArtifactMetadata)=>void }){
  const [name, setName] = useState<string>("*");
  const [types, setTypes] = useState<Record<ArtifactType, boolean>>({ model:true, dataset:true, code:true });
  const [items, setItems] = useState<ArtifactMetadata[]>([]);
  const [busy, setBusy] = useState(false);

  async function run(){
    setBusy(true);
    try{
      const q: ArtifactQuery = { name };
      const t = Object.entries(types).filter(([k,v])=>v).map(([k])=>k as ArtifactType);
      if (t.length>0 && t.length<3) q.types = t;
      const res = await api<ArtifactMetadata[]>("/artifacts?offset=1", { method:"POST", token, body: [q] });
      setItems(res);
    }catch(err:any){ onError(err.message || String(err)); }
    finally{ setBusy(false); }
  }

  return (
    <Card title="Search artifacts" actions={<Button onClick={run} disabled={busy}>{busy?"Searching…":"Search"}</Button>}>
      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <FieldLabel id="q">Name ("*" to enumerate)</FieldLabel>
          <Input id="q" value={name} onChange={e=>setName(e.target.value)} placeholder="audience-classifier or *" />
        </div>
        <fieldset className="mt-1 rounded-xl border p-3" aria-label="Filter by types">
          <legend className="text-sm font-medium">Types</legend>
          {(["model","dataset","code"] as ArtifactType[]).map(t=> (
            <label key={t} className="mr-4 inline-flex items-center gap-2">
              <input type="checkbox" checked={types[t]} onChange={e=>setTypes(prev=>({ ...prev, [t]: e.target.checked }))}/>
              <span>{t}</span>
            </label>
          ))}
        </fieldset>
      </div>
      <ul className="mt-4 divide-y rounded-xl border bg-white" role="list">
        {items.map(m=> (
          <li key={`${m.type}:${m.id}`} className="flex items-center justify-between px-4 py-3">
            <div>
              <div className="font-medium">{m.name}</div>
              <div className="text-sm text-gray-600">{m.type} · id: {m.id}</div>
            </div>
            <div className="flex gap-2">
              <Button className="bg-white border text-gray-900 hover:bg-gray-100" onClick={()=>onPick(m)}>Open</Button>
            </div>
          </li>
        ))}
        {items.length===0 && <li className="px-4 py-3 text-sm text-gray-600">No results.</li>}
      </ul>
    </Card>
  );
}

function Detail({ token, meta, onError, onOpen }:{ token:string; meta:ArtifactMetadata|null; onError:(m:string)=>void; onOpen:(v: "rate"|"lineage"|"license")=>void }){
  const [artifact, setArtifact] = useState<Artifact | null>(null);
  const [busy, setBusy] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(()=>{ (async()=>{
    if (!meta) return;
    setBusy(true);
    try{
      const res = await api<Artifact>(`/artifacts/${meta.type}/${meta.id}`, { token });
      setArtifact(res);
    }catch(err:any){ onError(err.message || String(err)); }
    finally{ setBusy(false); }
  })(); }, [meta?.id]);

  if (!meta) return <Card title="Artifact"><p>Select an artifact from Search.</p></Card>;

  return (
    <Card title={`Artifact: ${meta.name}`} actions={
      <>
        {meta.type === "model" && <>
          <Button className="bg-white border text-gray-900 hover:bg-gray-100" onClick={()=>onOpen("rate")}>Ratings</Button>
          <Button className="bg-white border text-gray-900 hover:bg-gray-100" onClick={()=>onOpen("lineage")}>Lineage</Button>
          <Button className="bg-white border text-gray-900 hover:bg-gray-100" onClick={()=>onOpen("license")}>License check</Button>
        </>}
        <Button className="bg-red-600 hover:bg-red-700" disabled={deleting} onClick={async()=>{
          if (!confirm("Delete this artifact? (NON-BASELINE)")) return;
          setDeleting(true);
          try{
            await api(`/artifact/${meta.type}/${meta.id}`, { method:"DELETE", token });
            alert("Deleted. Go back to search.");
          }catch(err:any){ alert(err.message || String(err)); }
          finally{ setDeleting(false); }
        }}>Delete</Button>
      </>
    }>
      {busy && <p>Loading…</p>}
      {artifact && (
        <div className="grid gap-2">
          <div><span className="font-semibold">Type:</span> {artifact.metadata.type}</div>
          <div><span className="font-semibold">ID:</span> {artifact.metadata.id}</div>
          <div><span className="font-semibold">Source URL:</span> <a className="underline" href={artifact.data.url} target="_blank" rel="noreferrer">{artifact.data.url}</a></div>
        </div>
      )}
    </Card>
  );
}

function Create({ token, onError, onCreated }:{ token:string; onError:(m:string)=>void; onCreated:(a:Artifact)=>void }){
  const [type, setType] = useState<ArtifactType>("model");
  const [url, setUrl] = useState("https://huggingface.co/google-bert/bert-base-uncased");
  const [busy, setBusy] = useState(false);
  async function submit(e: FormEvent){
    e.preventDefault();
    setBusy(true);
    try{
      const res = await api<Artifact>(`/artifact/${type}`, { method:"POST", token, body: { url } });
      onCreated(res);
    }catch(err:any){ onError(err.message || String(err)); }
    finally{ setBusy(false); }
  }
  return (
    <Card title="Register new artifact">
      <form onSubmit={submit} className="grid gap-3">
        <div>
          <FieldLabel id="t">Type</FieldLabel>
          <select id="t" value={type} onChange={e=>setType(e.target.value as ArtifactType)} className="mt-1 w-full rounded-xl border px-3 py-2 focus-visible:ring-2 focus-visible:ring-blue-600">
            <option value="model">model</option>
            <option value="dataset">dataset</option>
            <option value="code">code</option>
          </select>
        </div>
        <div>
          <FieldLabel id="u">Source URL</FieldLabel>
          <Input id="u" type="url" required value={url} onChange={e=>setUrl(e.target.value)} placeholder="https://…" />
        </div>
        <Button disabled={busy} type="submit">{busy?"Registering…":"Register"}</Button>
      </form>
    </Card>
  );
}

function Rate({ token, id }:{ token:string; id:string }){
  const [rating, setRating] = useState<ModelRating | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(()=>{ (async()=>{
    try{ setRating(await api<ModelRating>(`/artifact/model/${id}/rate`, { token })); }catch(e:any){ setError(e.message||String(e)); }
  })(); }, [id]);
  return (
    <Card title="Model rating">
      {error && <div role="alert" className="mb-3 rounded-md bg-red-50 p-3 text-red-900">{error}</div>}
      {!rating ? <p>Loading…</p> : (
        <div className="grid gap-2 md:grid-cols-2">
          <Metric label="Net score" value={rating.net_score}/>
          <Metric label="Ramp-up" value={rating.ramp_up_time}/>
          <Metric label="Bus factor" value={rating.bus_factor}/>
          <Metric label="Claims" value={rating.performance_claims}/>
          <Metric label="License" value={rating.license}/>
          <Metric label="Dataset+Code" value={rating.dataset_and_code_score}/>
          <Metric label="Dataset quality" value={rating.dataset_quality}/>
          <Metric label="Code quality" value={rating.code_quality}/>
          <Metric label="Reproducibility" value={rating.reproducibility}/>
          <Metric label="Reviewedness" value={rating.reviewedness}/>
          <Metric label="Tree score" value={rating.tree_score}/>
          <div className="rounded-xl border p-3"><div className="font-medium">Size score</div>
            <ul className="text-sm text-gray-700">
              <li>Raspberry Pi: {rating.size_score.raspberry_pi}</li>
              <li>Jetson Nano: {rating.size_score.jetson_nano}</li>
              <li>Desktop: {rating.size_score.desktop_pc}</li>
              <li>AWS: {rating.size_score.aws_server}</li>
            </ul>
          </div>
        </div>
      )}
    </Card>
  );
}
function Metric({ label, value }:{ label:string; value:number }){
  return <div className="rounded-xl border p-3"><div className="text-sm text-gray-600">{label}</div><div className="text-xl font-semibold">{value.toFixed(2)}</div></div>;
}

function Lineage({ token, id }:{ token:string; id:string }){
  const [g, setG] = useState<ArtifactLineageGraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(()=>{ (async()=>{
    try{ setG(await api<ArtifactLineageGraph>(`/artifact/model/${id}/lineage`, { token })); }catch(e:any){ setError(e.message||String(e)); }
  })(); }, [id]);
  return (
    <Card title="Lineage graph">
      {error && <div role="alert" className="mb-3 rounded-md bg-red-50 p-3 text-red-900">{error}</div>}
      {!g ? <p>Loading…</p> : (
        <div className="grid gap-3 md:grid-cols-2">
          <div>
            <div className="font-medium">Nodes</div>
            <ul className="mt-1 list-disc pl-5 text-sm text-gray-700">
              {g.nodes.map(n=> <li key={n.artifact_id}><span className="font-mono">{n.artifact_id}</span> — {n.name} <em className="text-gray-500">({n.source})</em></li>)}
            </ul>
          </div>
          <div>
            <div className="font-medium">Edges</div>
            <ul className="mt-1 list-disc pl-5 text-sm text-gray-700">
              {g.edges.map((e,i)=> <li key={i}><span className="font-mono">{e.from_node_artifact_id}</span> → <span className="font-mono">{e.to_node_artifact_id}</span> <em className="text-gray-500">({e.relationship})</em></li>)}
            </ul>
          </div>
        </div>
      )}
    </Card>
  );
}

function LicenseCheck({ token, id }:{ token:string; id:string }){
  const [repo, setRepo] = useState("https://github.com/google-research/bert");
  const [result, setResult] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  return (
    <Card title="License compatibility">
      <form className="grid gap-3" onSubmit={async (e)=>{
        e.preventDefault(); setBusy(true);
        try{ const ok = await api<boolean>(`/artifact/model/${id}/license-check`, { method:"POST", token, body:{ github_url: repo } }); setResult(ok); }
        catch(e:any){ alert(e.message||String(e)); }
        finally{ setBusy(false); }
      }}>
        <div>
          <FieldLabel id="g">GitHub URL</FieldLabel>
          <Input id="g" type="url" required value={repo} onChange={e=>setRepo(e.target.value)} />
        </div>
        <Button disabled={busy} type="submit">{busy?"Checking…":"Check"}</Button>
        {result!==null && <div className="rounded-xl border p-3">Compatibility: <span className={result?"text-green-700":"text-red-700"}>{String(result)}</span></div>}
      </form>
    </Card>
  );
}

function RegexSearch({ token, onPick }:{ token:string; onPick:(m:ArtifactMetadata)=>void }){
  const [regex, setRegex] = useState(".*?(audience|bert).*");
  const [items, setItems] = useState<ArtifactMetadata[]>([]);
  return (
    <Card title="Search by RegEx" actions={<Button onClick={async()=>{
      try{ const res = await api<ArtifactMetadata[]>("/artifact/byRegEx", { method:"POST", token, body:{ regex } }); setItems(res); }
      catch(e:any){ alert(e.message||String(e)); }
    }}>Search</Button>}>
      <div>
        <FieldLabel id="r">Regular expression</FieldLabel>
        <Input id="r" value={regex} onChange={e=>setRegex(e.target.value)} />
      </div>
      <ul className="mt-3 divide-y rounded-xl border">{items.map(m=> (
        <li key={`${m.type}:${m.id}`} className="flex items-center justify-between px-4 py-3">
          <div><div className="font-medium">{m.name}</div><div className="text-sm text-gray-600">{m.type} · {m.id}</div></div>
          <Button className="bg-white border text-gray-900 hover:bg-gray-100" onClick={()=>onPick(m)}>Open</Button>
        </li>
      ))}</ul>
    </Card>
  );
}

function ByName({ token, onPick }:{ token:string; onPick:(m:ArtifactMetadata)=>void }){
  const [name, setName] = useState("audience-classifier");
  const [items, setItems] = useState<ArtifactMetadata[]>([]);
  return (
    <Card title="Search by exact name" actions={<Button onClick={async()=>{
      try{ const res = await api<ArtifactMetadata[]>(`/artifact/byName/${encodeURIComponent(name)}`, { token }); setItems(res); }
      catch(e:any){ alert(e.message||String(e)); }
    }}>Search</Button>}>
      <div>
        <FieldLabel id="n">Name</FieldLabel>
        <Input id="n" value={name} onChange={e=>setName(e.target.value)} />
      </div>
      <ul className="mt-3 divide-y rounded-xl border">{items.map(m=> (
        <li key={`${m.type}:${m.id}`} className="flex items-center justify-between px-4 py-3">
          <div><div className="font-medium">{m.name}</div><div className="text-sm text-gray-600">{m.type} · {m.id}</div></div>
          <Button className="bg-white border text-gray-900 hover:bg-gray-100" onClick={()=>onPick(m)}>Open</Button>
        </li>
      ))}</ul>
    </Card>
  );
}
