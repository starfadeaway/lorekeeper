import { useEffect, useRef, useState } from 'react'
import { FALLBACK_GAMES, postDeliverablePlan, streamDeliverableApprove } from '../api'
import { download, fileStem } from '../exportMd'
import type {
  DeliverableConflict,
  DeliverablePlanDraft,
  DeliverablePlanItem,
  DeliverableResponse,
  DeliverableSection,
  Health,
} from '../types'

/**
 * 审批勾选 → 实际执行的计划。
 *
 * 默认全选（`chosen` 里没有的 key 视为选中）：用户只取消不想跑的，
 * 不需要逐条勾一遍。空计划由 UI 挡在按钮上，服务端也会再校验一次。
 */
export function pickPlan(
  plan: DeliverablePlanItem[],
  chosen: Record<string, boolean>,
): DeliverablePlanItem[] {
  return plan.filter((p) => chosen[p.name] !== false)
}

interface Props {
  health: Health | null
}

/** 子任务状态 → 展示文案与视觉档位。
 *
 *  四种状态必须**分开显示**，不能都归到「失败」或「无内容」：
 *  - empty     = 召回到了但抽不出可核对的事实（推荐类子任务常见）
 *  - truncated = 模型输出超预算被截断，本来能抽到（基础设施问题）
 *  - error     = 调用炸了
 *  混在一起会让人以为「知识库里没有」，那是**方向相反**的结论。
 */
const STATUS_VIEW: Record<string, { text: string; cls: string }> = {
  ok: { text: '完成', cls: 'dv-ok' },
  empty: { text: '无可用事实', cls: 'dv-empty' },
  truncated: { text: '输出截断', cls: 'dv-warn' },
  error: { text: '调用失败', cls: 'dv-bad' },
}

const SAMPLES = [
  '整理三月七的养成指南',
  '整理三月七的星魂效果',
  '整理三月七•存护的突破材料与行迹',
]

/**
 * 交付物标题 → 下载文件名（不含扩展名）与 `download()` 都搬到了
 * `../exportMd`，和问答导出的 Markdown 共用同一套规则。
 *
 * 两边各写一份的后果已经出现过：交付物这边原本用 run_id 当文件名，
 * 问答那边再写一次就会把这个 bug 复制一遍。现在改一处即两处生效。
 */

/** 冲突块。**不提供任何「挑一个」的交互** —— 见 DeliverableConflict 的注释。 */
function ConflictBlock({ c }: { c: DeliverableConflict }) {
  return (
    <div className="dv-conflict">
      <div className="dv-conflict-head">
        {c.subject} · {c.label} <code>{c.key}</code>
      </div>
      <table className="dv-table">
        <thead>
          <tr>
            <th>取值</th>
            <th>来源页面</th>
            <th>页面修订</th>
            <th>子任务</th>
          </tr>
        </thead>
        <tbody>
          {c.variants.map((v, i) => (
            <tr key={`${v.source_url}-${v.value}-${i}`}>
              <td>
                <b>
                  {v.value}
                  {v.unit ? ` ${v.unit}` : ''}
                </b>
              </td>
              <td>
                {v.source_url ? (
                  <a href={v.source_url} target="_blank" rel="noreferrer">
                    {v.page_title || v.source_url}
                  </a>
                ) : (
                  <span className="muted">{v.page_title || '—'}</span>
                )}
              </td>
              <td className="dv-rev">{v.revision || '—'}</td>
              <td className="muted">{v.subtask || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="dv-verdict">{c.verdict}</p>
    </div>
  )
}

function SectionBlock({ s }: { s: DeliverableSection }) {
  const view = STATUS_VIEW[s.status] ?? { text: s.status, cls: 'dv-empty' }
  const [open, setOpen] = useState(true)
  return (
    <section className="dv-section">
      <div className="dv-section-head" onClick={() => setOpen((v) => !v)}>
        <span className={`dv-badge ${view.cls}`}>{view.text}</span>
        <b>{s.name}</b>
        <span className="muted">
          {s.facts.length} 条事实 · {s.latency_ms}ms
        </span>
      </div>
      <div className="muted dv-goal">目标：{s.goal}</div>
      {s.queries.length > 0 && (
        <div className="chips">
          {s.queries.map((q) => (
            <span className="chip" key={q}>
              {q}
            </span>
          ))}
        </div>
      )}
      {s.notes && <div className="dv-note">{s.notes}</div>}
      {open && s.facts.length > 0 && (
        <table className="dv-table">
          <thead>
            <tr>
              <th>字段</th>
              <th>取值</th>
              <th>单位</th>
              <th>来源</th>
              <th>页面修订</th>
            </tr>
          </thead>
          <tbody>
            {s.facts.map((f, i) => (
              <tr key={`${f.key}-${f.value}-${i}`}>
                <td>{f.label}</td>
                <td className="dv-value">{f.value}</td>
                <td className="muted">{f.unit || '—'}</td>
                <td>
                  {f.source_url ? (
                    <a href={f.source_url} target="_blank" rel="noreferrer">
                      {f.page_title}
                    </a>
                  ) : (
                    <span className="muted">{f.page_title}</span>
                  )}
                </td>
                <td className="dv-rev">{f.revision || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}

/**
 * 交付物面板：一次请求产出多份结构化交付物。
 *
 * 与问答面板的根本差别不是「更长的回答」，而是**产出形态**：
 * 问答给一段文本，这里给结构化事实表 + 冲突标注 + 出处，
 * 并且 Markdown 与 JSON 同时到手 —— 前端渲染的是结构化那份，
 * Markdown 原文作为可复制/可下载的对照物。
 */
export function DeliverablePanel({ health }: Props) {
  const [question, setQuestion] = useState(SAMPLES[0])
  const [game, setGame] = useState('sr')
  const [busy, setBusy] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [res, setRes] = useState<DeliverableResponse | null>(null)
  const [draft, setDraft] = useState<DeliverablePlanDraft | null>(null)
  const [chosen, setChosen] = useState<Record<string, boolean>>({})
  /** 审批执行期间每个子任务的实时状态（来自 SSE） */
  const [progress, setProgress] = useState<
    Record<string, { status: string; facts?: number; ms?: number }>
  >({})
  const [err, setErr] = useState<string | null>(null)
  const [view, setView] = useState<'structured' | 'markdown'>('structured')
  const timer = useRef<number | null>(null)

  useEffect(() => () => { if (timer.current) window.clearInterval(timer.current) }, [])

  const stopTimer = () => {
    if (timer.current) window.clearInterval(timer.current)
    timer.current = null
  }

  /** 第一步：只出计划（一次规划调用，不检索）。 */
  const run = async () => {
    if (busy || question.trim().length < 4) return
    setBusy(true)
    setErr(null)
    setRes(null)
    setDraft(null)
    setElapsed(0)
    timer.current = window.setInterval(() => setElapsed((e) => e + 1), 1000)
    try {
      const d = await postDeliverablePlan({ question: question.trim(), game })
      setDraft(d)
      setChosen(Object.fromEntries(d.plan.map((p) => [p.name, true])))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      stopTimer()
      setBusy(false)
    }
  }

  /** 第二步：按勾选后的计划执行，用 SSE 实时更新每个子任务的状态。 */
  const approve = async () => {
    if (!draft || busy) return
    const plan = pickPlan(draft.plan, chosen)
    if (plan.length === 0) {
      setErr('至少保留一个子任务再批准')
      return
    }
    setBusy(true)
    setErr(null)
    setElapsed(0)
    setProgress(Object.fromEntries(plan.map((p) => [p.name, { status: 'pending' }])))
    timer.current = window.setInterval(() => setElapsed((e) => e + 1), 1000)
    try {
      const r = await streamDeliverableApprove(draft.run_id, plan, (ev) => {
        if (ev.event === 'subtask_start') {
          setProgress((prev) => ({ ...prev, [ev.data.name]: { status: 'running' } }))
        } else if (ev.event === 'subtask_end') {
          setProgress((prev) => ({
            ...prev,
            [ev.data.name]: { status: ev.data.status, facts: ev.data.facts, ms: ev.data.ms },
          }))
        }
      })
      setRes(r)
      setDraft(null)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      stopTimer()
      setBusy(false)
    }
  }

  const games = Object.entries(health?.games ?? FALLBACK_GAMES)
  const counts = res?.status_counts ?? {}
  const limits = (res?.json.meta?.known_limits as string) || ''

  return (
    <div className="retrieve dv">
      <div className="group-title">
        交付物任务
        <span className="muted dv-sub">
          一次调用产出多份结构化交付物（Markdown + JSON），每条事实带出处与页面修订
        </span>
      </div>

      <div className="controls">
        <textarea
          className="dv-input"
          rows={2}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="例如：整理三月七的养成指南"
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) void run()
          }}
        />
        <div className="row">
          <div className="segmented">
            {games.map(([slug, name]) => (
              <button
                key={slug}
                className={game === slug ? 'seg active' : 'seg'}
                onClick={() => setGame(slug)}
              >
                {name}
              </button>
            ))}
          </div>
          <button className="primary" onClick={run} disabled={busy || question.trim().length < 4}>
            {busy ? `处理中… ${elapsed}s` : '生成交付物'}
          </button>
        </div>
        <div className="chips">
          {SAMPLES.map((s) => (
            <span className="chip clickable" key={s} onClick={() => setQuestion(s)}>
              {s}
            </span>
          ))}
          <span className="muted dv-hint">Ctrl/⌘ + Enter 直接跑</span>
        </div>
      </div>

      {busy && (
        <div className="thinking dv-thinking">
          <span className="spinner" />{' '}
          {draft ? '按已批准的计划检索、抽取事实…' : '规划子任务…'}
          <span className="muted">
            已用 {elapsed}s{draft ? '（执行阶段 30–60s，属正常）' : '（只做一次规划调用）'}
          </span>
        </div>
      )}

      {/* 审批执行的实时进度：状态来自 SSE，与最终结果里的 sections 是同一份数据 */}
      {draft && busy && Object.keys(progress).length > 0 && (
        <div className="dv-progress">
          {draft.plan
            .filter((p) => progress[p.name])
            .map((p) => {
              const st = progress[p.name]
              const label =
                st.status === 'pending' ? '等待'
                : st.status === 'running' ? '执行中…'
                : st.status === 'ok' ? `完成 · ${st.facts ?? 0} 条事实`
                : st.status === 'empty' ? '无可用事实'
                : st.status === 'truncated' ? '输出截断'
                : '失败'
              return (
                <div className="dv-prog-item" data-state={st.status} key={p.name}>
                  <span className="dv-prog-dot" />
                  <span className="dv-prog-name">{p.name}</span>
                  <span className="muted dv-prog-meta">
                    {label}
                    {st.ms != null ? ` · ${st.ms}ms` : ''}
                  </span>
                </div>
              )
            })}
        </div>
      )}

      {err && <div className="error">出错了：{err}</div>}

      {draft && !busy && (
        <div className="dv-plan">
          <div className="group-title">
            审批：确认要执行的子任务
            <span className="muted dv-sub">
              默认全选；取消勾选可跳过某条。批准后只跑保留项，不会重新规划。
            </span>
          </div>
          <p className="muted">
            主题：{draft.subject || '未识别'} · run <code>{draft.run_id}</code> · 规划耗时{' '}
            {draft.elapsed_ms}ms
          </p>
          {draft.plan.map((p) => (
            <label className="dv-plan-item" key={p.name}>
              <input
                type="checkbox"
                checked={chosen[p.name] !== false}
                onChange={(e) =>
                  setChosen((c) => ({ ...c, [p.name]: e.target.checked }))
                }
              />
              <span className="dv-plan-body">
                <b>{p.name}</b>
                <span className="muted"> {p.goal}</span>
                <span className="chips">
                  {p.queries.map((q) => (
                    <span className="chip" key={q}>
                      {q}
                    </span>
                  ))}
                </span>
              </span>
            </label>
          ))}
          <div className="row dv-plan-actions">
            <span className="muted">
              将执行 {pickPlan(draft.plan, chosen).length} / {draft.plan.length} 个子任务
            </span>
            <button className="ghost mini" onClick={() => setDraft(null)}>
              取消
            </button>
            <button
              className="primary"
              onClick={approve}
              disabled={busy || pickPlan(draft.plan, chosen).length === 0}
            >
              批准并执行
            </button>
          </div>
        </div>
      )}

      {res && (
        <div className="dv-result">
          <div className="metrics">
            <div className="stat">
              <span className="muted">事实</span>
              <b>{res.json.sections.reduce((a, s) => a + s.facts.length, 0)}</b>
            </div>
            <div className="stat">
              <span className="muted">子任务</span>
              <b>
                {counts.subtasks_ok ?? 0}/{res.json.sections.length}
              </b>
            </div>
            <div className="stat">
              <span className="muted">冲突</span>
              <b>{res.conflicts_total}</b>
            </div>
            <div className="stat">
              <span className="muted">耗时</span>
              <b>{(res.elapsed_ms / 1000).toFixed(1)}s</b>
            </div>
            <div className="stat">
              <span className="muted">出处</span>
              <b>{res.json.refs.length}</b>
            </div>
          </div>

          <div className="row dv-actions">
            <div className="segmented">
              <button
                className={view === 'structured' ? 'seg active' : 'seg'}
                onClick={() => setView('structured')}
              >
                结构化
              </button>
              <button
                className={view === 'markdown' ? 'seg active' : 'seg'}
                onClick={() => setView('markdown')}
              >
                Markdown 原文
              </button>
            </div>
            <button
              className="ghost mini"
              onClick={() => navigator.clipboard?.writeText(res.markdown)}
            >
              复制 Markdown
            </button>
            <button
              className="ghost mini"
              onClick={() =>
                download(
                  `${fileStem(res.json.topic, res.json.run_id)}.md`,
                  res.markdown,
                  'text/markdown;charset=utf-8',
                )
              }
            >
              下载 .md
            </button>
            <button
              className="ghost mini"
              onClick={() =>
                download(
                  `${fileStem(res.json.topic, res.json.run_id)}.json`,
                  JSON.stringify(res.json, null, 2),
                  'application/json;charset=utf-8',
                )
              }
            >
              下载 .json
            </button>
          </div>

          {view === 'markdown' ? (
            // 后端渲染的原文，前端不再二次解析 Markdown ——
            // 两套 Markdown 渲染器迟早会在某个语法上对不齐，
            // 而那种 bug 极难定位（看起来像内容错，其实是渲染错）。
            <pre className="dv-md">{res.markdown}</pre>
          ) : (
            <>
              <h3 className="dv-h3">{res.json.topic}</h3>
              <p className="muted">
                知识库 {res.json.game} · 生成于 {res.json.generated_at} · run{' '}
                <code>{res.json.run_id}</code>
              </p>

              {res.json.conflicts.length > 0 && (
                <div className="dv-conflicts">
                  <div className="group-title">
                    检测到 {res.json.conflicts.length} 处数据冲突
                  </div>
                  {res.json.conflicts.map((c) => (
                    <ConflictBlock key={`${c.subject}-${c.key}`} c={c} />
                  ))}
                </div>
              )}

              {res.json.sections.map((s) => (
                <SectionBlock key={s.name} s={s} />
              ))}

              {limits && <div className="dv-limits">已知限制：{limits}</div>}

              {res.json.sections.some((s) => s.offtopic_dropped > 0) && (
                <div className="dv-limits">
                  主题闸门：共拦下{' '}
                  {res.json.sections.reduce(
                    (n, s) => n + s.offtopic_dropped,
                    0,
                  )}{' '}
                  条（逐子任务明细见上方各卡片）
                </div>
              )}

              {res.json.refs.length > 0 && (
                <div className="dv-refs">
                  <div className="group-title">出处</div>
                  {res.json.refs.map((r) => (
                    <div key={r.url}>
                      <a href={r.url} target="_blank" rel="noreferrer">
                        {r.title || r.url}
                      </a>
                      {r.last_revision_ts && (
                        <span className="muted dv-rev"> 页面修订 {r.last_revision_ts}</span>
                      )}
                    </div>
                  ))}
                  <p className="muted dv-limits">
                    「页面修订」是 wiki 页面最后一次被编辑的时间，**不是数据生效日期**。
                    本页两年没被动过通常意味着 wiki 认为它不需要改，而不是内容已过时。
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
