// WebGL2 instanced renderer for the board's bulk geometry: region circles, density aggregates,
// nodes and edges. One draw call per layer; nothing here allocates per frame.

export type RGBA = number; // 0xRRGGBBAA

const SHAPE_F = 8; // floats per shape instance
const SHAPE_C = 12; // colour bytes per shape instance
const LINE_F = 6;
const LINE_C = 4;

export class ShapeBatch {
  f: Float32Array;
  c: Uint8Array;
  n = 0;
  constructor(cap = 256) {
    this.f = new Float32Array(cap * SHAPE_F);
    this.c = new Uint8Array(cap * SHAPE_C);
  }
  private grow() {
    const f = new Float32Array(this.f.length * 2);
    f.set(this.f);
    this.f = f;
    const c = new Uint8Array(this.c.length * 2);
    c.set(this.c);
    this.c = c;
  }
  /**
   * shape: 0 circle, 1 square. r in world units, clamped to [minPx, maxPx] on screen (maxPx 0 = no limit).
   * strokeW/ringW in px; ring is drawn outside the shape with a small gap. segs > 1 draws cell dividers.
   * dash > 0 dashes the stroke with that dash length in px.
   */
  push(x: number, y: number, r: number, minPx: number, maxPx: number, shape: number, fill: RGBA, stroke: RGBA = 0, strokeW = 0, ring: RGBA = 0, ringW = 0, segs = 0, dash = 0) {
    if ((this.n + 1) * SHAPE_F > this.f.length) this.grow();
    let i = this.n * SHAPE_F;
    const f = this.f;
    f[i++] = x;
    f[i++] = y;
    f[i++] = r;
    f[i++] = minPx;
    f[i++] = maxPx;
    f[i++] = shape + segs * 4; // packed: shape in the low 2 bits
    f[i++] = strokeW + (dash > 0 ? 64 + dash * 256 : 0); // packed: dash length above the stroke width
    f[i++] = ringW;
    let j = this.n * SHAPE_C;
    const c = this.c;
    c[j++] = fill >>> 24;
    c[j++] = (fill >>> 16) & 255;
    c[j++] = (fill >>> 8) & 255;
    c[j++] = fill & 255;
    c[j++] = stroke >>> 24;
    c[j++] = (stroke >>> 16) & 255;
    c[j++] = (stroke >>> 8) & 255;
    c[j++] = stroke & 255;
    c[j++] = ring >>> 24;
    c[j++] = (ring >>> 16) & 255;
    c[j++] = (ring >>> 8) & 255;
    c[j++] = ring & 255;
    this.n++;
  }
}

export class LineBatch {
  f: Float32Array;
  c: Uint8Array;
  n = 0;
  constructor(cap = 256) {
    this.f = new Float32Array(cap * LINE_F);
    this.c = new Uint8Array(cap * LINE_C);
  }
  push(x0: number, y0: number, x1: number, y1: number, width: number, color: RGBA, arrow = 0) {
    if ((this.n + 1) * LINE_F > this.f.length) {
      const f = new Float32Array(this.f.length * 2);
      f.set(this.f);
      this.f = f;
      const c = new Uint8Array(this.c.length * 2);
      c.set(this.c);
      this.c = c;
    }
    let i = this.n * LINE_F;
    this.f[i++] = x0;
    this.f[i++] = y0;
    this.f[i++] = x1;
    this.f[i++] = y1;
    this.f[i++] = width;
    this.f[i++] = arrow;
    let j = this.n * LINE_C;
    this.c[j++] = color >>> 24;
    this.c[j++] = (color >>> 16) & 255;
    this.c[j++] = (color >>> 8) & 255;
    this.c[j++] = color & 255;
    this.n++;
  }
}

const SHAPE_VS = `#version 300 es
layout(location=0) in vec2 a_corner;
layout(location=1) in vec2 a_pos;
layout(location=2) in vec4 a_size;   // r world, min px, max px, shape+segs*4
layout(location=3) in vec2 a_misc;   // stroke width (+dash packing), ring width
layout(location=4) in vec4 a_fill;
layout(location=5) in vec4 a_stroke;
layout(location=6) in vec4 a_ring;
uniform vec2 u_center;
uniform float u_zoom;
uniform vec2 u_res;
out vec2 v_p;
flat out float v_r;
flat out float v_shape;
flat out float v_segs;
flat out float v_sw;
flat out float v_dash;
flat out float v_rw;
flat out vec4 v_fill;
flat out vec4 v_stroke;
flat out vec4 v_ring;
void main() {
  float r = max(a_size.x * u_zoom, a_size.y);
  if (a_size.z > 0.0) r = min(r, a_size.z);
  float packed = a_size.w;
  v_segs = floor(packed / 4.0);
  v_shape = packed - v_segs * 4.0;
  float sw = a_misc.x;
  v_dash = 0.0;
  if (sw >= 64.0) { v_dash = floor(sw / 256.0); sw = sw - v_dash * 256.0 - 64.0; }
  v_sw = sw;
  v_rw = a_misc.y;
  float pad = v_rw > 0.0 ? 3.0 + v_rw : 0.0;
  float h = r + pad + 1.5;
  vec2 c = (a_pos - u_center) * u_zoom;
  vec2 p = c + a_corner * h;
  gl_Position = vec4(p.x / (u_res.x * 0.5), -p.y / (u_res.y * 0.5), 0.0, 1.0);
  v_p = a_corner * h;
  v_r = r;
  v_fill = a_fill; v_stroke = a_stroke; v_ring = a_ring;
}`;

const SHAPE_FS = `#version 300 es
precision highp float;
in vec2 v_p;
flat in float v_r;
flat in float v_shape;
flat in float v_segs;
flat in float v_sw;
flat in float v_dash;
flat in float v_rw;
flat in vec4 v_fill;
flat in vec4 v_stroke;
flat in vec4 v_ring;
uniform float u_px;
uniform float u_alpha;
out vec4 o;
const float PI = 3.14159265;
void main() {
  bool sq = v_shape > 0.5;
  float rr = sq ? v_r * 0.9 : v_r;
  float d = sq ? max(abs(v_p.x), abs(v_p.y)) - rr : length(v_p) - rr;
  float aa = u_px;
  float inside = clamp(0.5 - d / aa, 0.0, 1.0);
  vec3 col = v_fill.rgb * v_fill.a * inside;
  float a = v_fill.a * inside;
  if (v_segs > 1.5 && v_r > 12.0 && d < -v_sw) {
    float dist;
    if (sq) {
      float hgt = 2.0 * rr / v_segs;
      float m = mod(v_p.y + rr, hgt);
      dist = min(m, hgt - m);
      if (abs(v_p.y) > rr - hgt * 0.5) dist = 99.0;
    } else {
      float w = 2.0 * PI / v_segs;
      float ang = atan(v_p.x, -v_p.y);
      float m = mod(ang, w);
      dist = min(m, w - m) * length(v_p);
    }
    float s = clamp(0.5 - (dist - 0.75) / aa, 0.0, 1.0);
    col = mix(col, vec3(0.0196) * 1.0, s);
    a = mix(a, 1.0, s);
  }
  if (v_sw > 0.0) {
    float band = clamp(0.5 - (abs(d + v_sw * 0.5) - v_sw * 0.5) / aa, 0.0, 1.0);
    if (v_dash > 0.0) {
      float arc = sq ? (v_p.x + v_p.y) : atan(v_p.y, v_p.x) * v_r;
      band *= step(0.5, fract(arc / (2.0 * v_dash)));
    }
    float s = band * v_stroke.a;
    col = col * (1.0 - s) + v_stroke.rgb * s;
    a = a * (1.0 - s) + s;
  }
  if (v_rw > 0.0) {
    float g = 2.5 + v_rw * 0.5;
    float band = clamp(0.5 - (abs(d - g) - v_rw * 0.5) / aa, 0.0, 1.0);
    float s = band * v_ring.a;
    col = col * (1.0 - s) + v_ring.rgb * s;
    a = a * (1.0 - s) + s;
  }
  o = vec4(col, a) * u_alpha;
}`;

const LINE_VS = `#version 300 es
layout(location=0) in vec2 a_corner;  // x 0..1 along, y -1..1 across
layout(location=1) in vec4 a_seg;
layout(location=2) in vec2 a_w;       // width px, arrow
layout(location=3) in vec4 a_color;
uniform vec2 u_center;
uniform float u_zoom;
uniform vec2 u_res;
out float v_across;
out float v_along;
flat out float v_len;
flat out float v_w;
flat out float v_arrow;
flat out vec4 v_color;
void main() {
  vec2 p0 = (a_seg.xy - u_center) * u_zoom;
  vec2 p1 = (a_seg.zw - u_center) * u_zoom;
  vec2 dir = p1 - p0;
  float len = max(length(dir), 0.0001);
  vec2 t = dir / len;
  vec2 n = vec2(-t.y, t.x);
  float h = a_w.x * 0.5 + 1.0 + a_w.y * 4.5;
  vec2 p = mix(p0, p1, a_corner.x) + n * a_corner.y * h;
  gl_Position = vec4(p.x / (u_res.x * 0.5), -p.y / (u_res.y * 0.5), 0.0, 1.0);
  v_across = a_corner.y * h;
  v_along = a_corner.x * len;
  v_len = len; v_w = a_w.x; v_arrow = a_w.y; v_color = a_color;
}`;

const LINE_FS = `#version 300 es
precision highp float;
in float v_across;
in float v_along;
flat in float v_len;
flat in float v_w;
flat in float v_arrow;
flat in vec4 v_color;
uniform float u_px;
uniform float u_alpha;
uniform float u_node_r;
out vec4 o;
void main() {
  float hw = v_w * 0.5;
  if (v_arrow > 0.5 && v_len > 2.0 * u_node_r + 16.0) {
    float e = v_len - u_node_r - 2.0 - v_along;   // distance before the target node's edge
    if (e < 0.0) discard;
    if (e < 10.0) hw = max(hw, e * 0.42);
  }
  float a = clamp(0.5 - (abs(v_across) - hw) / u_px, 0.0, 1.0) * v_color.a;
  o = vec4(v_color.rgb * a, a) * u_alpha;
}`;

interface Layer {
  kind: 'shape' | 'line';
  vao: WebGLVertexArrayObject;
  fbuf: WebGLBuffer;
  cbuf: WebGLBuffer;
  n: number;
  alpha: number;
  fcap: number;
  ccap: number;
}

export class Renderer {
  gl: WebGL2RenderingContext;
  private shapeProg!: WebGLProgram;
  private lineProg!: WebGLProgram;
  private quad!: WebGLBuffer;
  private lineQuad!: WebGLBuffer;
  private layers = new Map<string, Layer>();
  private uShape: Record<string, WebGLUniformLocation | null> = {};
  private uLine: Record<string, WebGLUniformLocation | null> = {};
  w = 1;
  h = 1;
  dpr = 1;

  constructor(public canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, depth: false, stencil: false, powerPreference: 'high-performance', preserveDrawingBuffer: false });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;
    this.init();
  }

  private compile(vs: string, fs: string): WebGLProgram {
    const gl = this.gl;
    const mk = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('Shader: ' + gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, mk(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('Program: ' + gl.getProgramInfoLog(p));
    return p;
  }

  init() {
    const gl = this.gl;
    this.layers.clear();
    this.shapeProg = this.compile(SHAPE_VS, SHAPE_FS);
    this.lineProg = this.compile(LINE_VS, LINE_FS);
    for (const n of ['u_center', 'u_zoom', 'u_res', 'u_px', 'u_alpha']) this.uShape[n] = gl.getUniformLocation(this.shapeProg, n);
    for (const n of ['u_center', 'u_zoom', 'u_res', 'u_px', 'u_alpha', 'u_node_r']) this.uLine[n] = gl.getUniformLocation(this.lineProg, n);
    this.quad = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    this.lineQuad = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineQuad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, -1, 1, -1, 0, 1, 1, 1]), gl.STATIC_DRAW);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  resize(w: number, h: number, dpr: number) {
    this.w = Math.max(1, w);
    this.h = Math.max(1, h);
    this.dpr = dpr;
    const cw = Math.round(this.w * dpr);
    const ch = Math.round(this.h * dpr);
    if (this.canvas.width !== cw || this.canvas.height !== ch) {
      this.canvas.width = cw;
      this.canvas.height = ch;
    }
  }

  private layer(name: string, kind: 'shape' | 'line'): Layer {
    let L = this.layers.get(name);
    if (L) return L;
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    const fbuf = gl.createBuffer()!;
    const cbuf = gl.createBuffer()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, kind === 'shape' ? this.quad : this.lineQuad);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, fbuf);
    if (kind === 'shape') {
      const stride = SHAPE_F * 4;
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, stride, 0);
      gl.vertexAttribDivisor(1, 1);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 4, gl.FLOAT, false, stride, 8);
      gl.vertexAttribDivisor(2, 1);
      gl.enableVertexAttribArray(3);
      gl.vertexAttribPointer(3, 2, gl.FLOAT, false, stride, 24);
      gl.vertexAttribDivisor(3, 1);
      gl.bindBuffer(gl.ARRAY_BUFFER, cbuf);
      for (let k = 0; k < 3; k++) {
        gl.enableVertexAttribArray(4 + k);
        gl.vertexAttribPointer(4 + k, 4, gl.UNSIGNED_BYTE, true, SHAPE_C, k * 4);
        gl.vertexAttribDivisor(4 + k, 1);
      }
    } else {
      const stride = LINE_F * 4;
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 4, gl.FLOAT, false, stride, 0);
      gl.vertexAttribDivisor(1, 1);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 2, gl.FLOAT, false, stride, 16);
      gl.vertexAttribDivisor(2, 1);
      gl.bindBuffer(gl.ARRAY_BUFFER, cbuf);
      gl.enableVertexAttribArray(3);
      gl.vertexAttribPointer(3, 4, gl.UNSIGNED_BYTE, true, LINE_C, 0);
      gl.vertexAttribDivisor(3, 1);
    }
    gl.bindVertexArray(null);
    L = { kind, vao, fbuf, cbuf, n: 0, alpha: 1, fcap: 0, ccap: 0 };
    this.layers.set(name, L);
    return L;
  }

  private upload(L: Layer, f: Float32Array, c: Uint8Array, n: number, fPer: number, cPer: number) {
    const gl = this.gl;
    const fv = f.subarray(0, n * fPer);
    const cv = c.subarray(0, n * cPer);
    gl.bindBuffer(gl.ARRAY_BUFFER, L.fbuf);
    if (fv.byteLength > L.fcap) {
      gl.bufferData(gl.ARRAY_BUFFER, f.byteLength, gl.DYNAMIC_DRAW);
      L.fcap = f.byteLength;
    }
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, fv);
    gl.bindBuffer(gl.ARRAY_BUFFER, L.cbuf);
    if (cv.byteLength > L.ccap) {
      gl.bufferData(gl.ARRAY_BUFFER, c.byteLength, gl.DYNAMIC_DRAW);
      L.ccap = c.byteLength;
    }
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, cv);
    L.n = n;
  }

  setShapes(name: string, b: ShapeBatch) {
    this.upload(this.layer(name, 'shape'), b.f, b.c, b.n, SHAPE_F, SHAPE_C);
  }
  setLines(name: string, b: LineBatch) {
    this.upload(this.layer(name, 'line'), b.f, b.c, b.n, LINE_F, LINE_C);
  }
  setAlpha(name: string, a: number) {
    const L = this.layers.get(name);
    if (L) L.alpha = a;
  }
  count(name: string) {
    return this.layers.get(name)?.n ?? 0;
  }

  draw(cx: number, cy: number, zoom: number, order: string[], nodeRpx: number) {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(5 / 255, 5 / 255, 5 / 255, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    let cur: WebGLProgram | null = null;
    for (const name of order) {
      const L = this.layers.get(name);
      if (!L || !L.n || L.alpha <= 0.003) continue;
      const prog = L.kind === 'shape' ? this.shapeProg : this.lineProg;
      const u = L.kind === 'shape' ? this.uShape : this.uLine;
      if (cur !== prog) {
        gl.useProgram(prog);
        cur = prog;
        gl.uniform2f(u.u_center, cx, cy);
        gl.uniform1f(u.u_zoom, zoom);
        gl.uniform2f(u.u_res, this.w, this.h);
        gl.uniform1f(u.u_px, 1 / this.dpr);
        if (L.kind === 'line') gl.uniform1f(u.u_node_r, nodeRpx);
      }
      gl.uniform1f(u.u_alpha, L.alpha);
      gl.bindVertexArray(L.vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, L.n);
    }
    gl.bindVertexArray(null);
  }
}
