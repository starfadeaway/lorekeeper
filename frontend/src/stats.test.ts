/**
 * 守卫：侧边栏的片段统计**只算生产索引**。
 *
 * 这条 bug 曾经靠 `tsc` 的类型不匹配偶然挡住（collection 缺 game/variant
 * 字段），但类型检查不是测试 —— 它只说明「形状不对」，不说明「算错了」。
 * 而这些数字会被拿去写简历，所以值得单独锁住。
 */
import { describe, expect, it } from 'vitest'

import {
  chunksPerGame,
  countExperimentChunks,
  countProductionChunks,
  isProductionCollection,
} from './stats'

const MODEL = 'BAAI/bge-m3'

const cols = [
  { name: 'kb_sr_baai-bge-m3_base', count: 3932, game: 'sr', variant: 'base', embed_model: MODEL },
  { name: 'kb_sr_baai-bge-m3_b1', count: 2808, game: 'sr', variant: 'b1', embed_model: MODEL },
  { name: 'kb_sr_baai-bge-m3_b2', count: 2238, game: 'sr', variant: 'b2', embed_model: MODEL },
  { name: 'kb_ys_baai-bge-m3_base', count: 5413, game: 'ys', variant: 'base', embed_model: MODEL },
  { name: 'kb_sr_Qwen_base', count: 1000, game: 'sr', variant: 'base', embed_model: 'Qwen/Qwen3' },
]

describe('isProductionCollection', () => {
  it('只认「当前生产模型 + base 变体」', () => {
    expect(isProductionCollection(cols[0], MODEL)).toBe(true)
    expect(isProductionCollection(cols[1], MODEL)).toBe(false) // b1 是实验变体
    expect(isProductionCollection(cols[4], MODEL)).toBe(false) // Qwen 不是生产模型
  })

  it('缺 variant 字段按 base 算（老索引没有这个字段）', () => {
    expect(isProductionCollection({ name: 'kb_sr_x', embed_model: MODEL }, MODEL)).toBe(true)
  })
})

describe('countProductionChunks', () => {
  it('变体库不得混进生产合计', () => {
    // 生产 = 3932(sr base) + 5413(ys base) = 9345
    // 若把 b1/b2 也算进来会变成 14391 —— 那正是「虚高 4.4 倍」的由来
    expect(countProductionChunks(cols, MODEL)).toBe(9345)
  })

  it('实验合计与生产合计不重叠、相加等于全部', () => {
    const prod = countProductionChunks(cols, MODEL)
    const exp = countExperimentChunks(cols, MODEL)
    const total = cols.reduce((s, c) => s + (c.count ?? 0), 0)
    expect(prod + exp).toBe(total)
    expect(exp).toBe(2808 + 2238 + 1000)
  })
})

describe('chunksPerGame', () => {
  it('按游戏分开统计，且只用生产库', () => {
    const per = chunksPerGame(cols, MODEL)
    expect(per.get('sr')).toBe(3932) // 不含 b1/b2
    expect(per.get('ys')).toBe(5413)
  })

  it('缺 game 字段时从集合名解析', () => {
    const per = chunksPerGame(
      [{ name: 'kb_zzz_baai-bge-m3_base', count: 1498, variant: 'base', embed_model: MODEL }],
      MODEL,
    )
    expect(per.get('zzz')).toBe(1498)
  })
})
