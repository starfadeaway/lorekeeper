import type { ToolCallView } from '../types'

const TOOL_LABEL: Record<string, string> = {
  search_kb: '检索知识库',
  get_page: '抓取页面',
  calculator: '计算',
  list_games: '查询知识库列表',
}

/**
 * 失败原因的单行摘要：折叠换行、截到 90 字。
 *
 * 只测纯函数（项目里其他组件测试也是这个口径）：渲染与网络交互需要 jsdom + mock，
 * 收益不足以支撑引那套依赖，而「截断/换行/空值」才是真正会出 bug 的位置。
 */
export function errorLine(error: string | null | undefined, max = 90): string {
  const one = String(error ?? '').replace(/\s+/g, ' ').trim()
  return one.length > max ? `${one.slice(0, max)}…` : one
}

/**
 * 工具调用时间线。
 * 这是整个界面最想让人看到的东西：Agent 不是黑箱，
 * 每一步用了什么工具、传了什么参数、多久返回、成没成功，全部摊开。
 */
export function ToolTimeline({ tools }: { tools: ToolCallView[] }) {
  if (tools.length === 0) return null
  return (
    <div className="tools-timeline">
      {tools.map((t, i) => (
        <div key={i} className="tl-item" data-state={t.running ? 'running' : t.ok ? 'ok' : 'fail'}>
          <span className="tl-dot" />
          <span className="tl-name">{TOOL_LABEL[t.name] ?? t.name}</span>
          <code className="tl-args">{summarize(t.args)}</code>
          <span className="tl-meta">
            {t.running ? (
              <span className="spinner" />
            ) : (
              <>
                {t.ms}ms{t.chars ? ` · ${t.chars} 字符` : ''}
              </>
            )}
          </span>
          {/* 失败时必须给出原因：只显示红点 + 字符数，用户没法判断是参数问题还是服务挂了 */}
          {!t.running && !t.ok && t.error ? (
            <span className="tl-error" title={String(t.error)}>
              {errorLine(t.error)}
            </span>
          ) : null}
        </div>
      ))}
    </div>
  )
}

function summarize(args: Record<string, unknown>): string {
  const keys = Object.keys(args)
  if (keys.length === 0) return '()'
  const main = keys
    .map((k) => {
      const v = String(args[k] ?? '')
      return `${k}=${v.length > 34 ? `${v.slice(0, 34)}…` : v}`
    })
    .join('  ')
  return `{ ${main} }`
}
