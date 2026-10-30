import { describe, expect, it } from 'vitest'

import { errorLine } from './ToolTimeline'

describe('errorLine', () => {
  it('折叠换行与多余空白，只留一行', () => {
    expect(errorLine('[检索失败] boom\n  stack\ttrace')).toBe('[检索失败] boom stack trace')
  })

  it('超过 90 字截断并加省略号', () => {
    const got = errorLine('x'.repeat(120))
    expect(got.length).toBe(91)
    expect(got.endsWith('…')).toBe(true)
  })

  it('空值返回空串（成功时后端不下发 error）', () => {
    expect(errorLine(null)).toBe('')
    expect(errorLine(undefined)).toBe('')
  })
})
