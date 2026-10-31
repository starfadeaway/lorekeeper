import { describe, expect, it } from 'vitest'

import { pickPlan } from './DeliverablePanel'

const plan = [
  { name: '甲', goal: 'g甲', queries: ['q甲'] },
  { name: '乙', goal: 'g乙', queries: ['q乙'] },
]

describe('pickPlan', () => {
  it('默认全选：chosen 里没有的 key 视为选中', () => {
    expect(pickPlan(plan, {})).toEqual(plan)
  })

  it('取消勾选只影响对应项，顺序不变', () => {
    expect(pickPlan(plan, { 乙: false }).map((p) => p.name)).toEqual(['甲'])
  })

  it('未知 key 被忽略，不产生额外项', () => {
    expect(pickPlan(plan, { 不存在: false })).toEqual(plan)
  })
})
