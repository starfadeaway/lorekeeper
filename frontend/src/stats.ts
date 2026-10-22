/**
 * 索引统计的纯函数 —— 从 `Sidebar.tsx` 里提出来，为的是能测。
 *
 * 背景：这里算出的数字**会被拿去写简历**。实验 B/D 会在同一游戏下建出
 * b1/b2/b3 与 qwen 等变体库，把它们混进生产统计会让「星铁 9,986 片段」
 * 这种数字出现——而生产 base 库只有 3932。这条 bug 此前靠 `tsc` 的类型
 * 不匹配偶然挡住，没有测试守着。
 */

export type CollectionLike = {
  name: string
  count?: number
  game?: string
  variant?: string
  embed_model?: string
}

/** 只有「当前生产模型 + base 变体」才算生产索引。 */
export function isProductionCollection(
  c: CollectionLike,
  productionModel: string | undefined,
): boolean {
  return (c.variant ?? 'base') === 'base' && c.embed_model === productionModel
}

/** 生产片段总数。 */
export function countProductionChunks(
  collections: CollectionLike[],
  productionModel: string | undefined,
): number {
  return collections
    .filter((c) => isProductionCollection(c, productionModel))
    .reduce((s, c) => s + (c.count ?? 0), 0)
}

/** 实验/变体库的片段总数（与生产分开统计，不并入任何对外数字）。 */
export function countExperimentChunks(
  collections: CollectionLike[],
  productionModel: string | undefined,
): number {
  return collections
    .filter((c) => !isProductionCollection(c, productionModel))
    .reduce((s, c) => s + (c.count ?? 0), 0)
}

/** 每个游戏的生产片段数；优先用后端下发的 game 字段，缺失才从集合名解析。 */
export function chunksPerGame(
  collections: CollectionLike[],
  productionModel: string | undefined,
): Map<string, number> {
  const per = new Map<string, number>()
  for (const c of collections) {
    if (!isProductionCollection(c, productionModel)) continue
    const m = /^kb_([^_]+)_/.exec(c.name)
    const game = c.game ?? m?.[1]
    if (game) per.set(game, (per.get(game) ?? 0) + (c.count ?? 0))
  }
  return per
}
