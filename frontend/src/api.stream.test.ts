import { afterEach, describe, expect, it, vi } from 'vitest'

import { streamDeliverableApprove } from './api'
import type { DeliverableProgressEvent } from './types'

const RESULT = {
  markdown: 'md',
  json: { run_id: 'r1' },
  status_counts: {},
  conflicts_total: 0,
  elapsed_ms: 1,
}

function responseFrom(frames: string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const enc = new TextEncoder()
      for (const f of frames) controller.enqueue(enc.encode(f))
      controller.close()
    },
  })
  return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
}

afterEach(() => vi.unstubAllGlobals())

describe('streamDeliverableApprove', () => {
  it('按顺序回调事件，并返回 result', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      responseFrom([
        'event: plan\ndata: {"topic":"t","subject":"s","plan":[],"reused":[]}\n\n',
        'event: subtask_start\ndata: {"name":"甲","index":1,"total":1}\n\n',
        'event: subtask_end\ndata: {"name":"甲","status":"ok","facts":2,"ms":10}\n\n',
        `event: result\ndata: ${JSON.stringify(RESULT)}\n\n`,
      ]),
    ))

    const seen: DeliverableProgressEvent[] = []
    const got = await streamDeliverableApprove('r1', [], (ev) => seen.push(ev))

    expect(got.markdown).toBe('md')
    expect(seen.map((e) => e.event)).toEqual(['plan', 'subtask_start', 'subtask_end'])
  })

  it('中文被切在多字节字符中间也能拼回', async () => {
    const frame = 'event: plan\ndata: {"topic":"养成","subject":"三月七","plan":[],"reused":[]}\n\n'
    const bytes = new TextEncoder().encode(frame)
    // 在 UTF-8 多字节字符中间切一刀：解码必须用 {stream:true}，否则会出替换符
    const cut = bytes.indexOf(new TextEncoder().encode('三')[0]) + 1
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, cut))
        controller.enqueue(bytes.slice(cut))
        controller.close()
      },
    })
    vi.stubGlobal('fetch', vi.fn(async () =>
      new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } }),
    ))

    const seen: DeliverableProgressEvent[] = []
    await expect(
      streamDeliverableApprove('r1', [], (ev) => seen.push(ev)),
    ).rejects.toThrow('流结束但没有收到 result 事件')

    expect(seen[0]).toMatchObject({ event: 'plan', data: { subject: '三月七' } })
  })

  it('error 帧抛出可读错误', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      responseFrom([
        'event: error\ndata: {"message":"该 run 已完成，不能再审批执行"}\n\n',
      ]),
    ))

    await expect(streamDeliverableApprove('r1', [], () => {})).rejects.toThrow(
      '该 run 已完成，不能再审批执行',
    )
  })
})
