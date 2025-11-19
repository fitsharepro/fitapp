"""Ferramentas para sugerir jogos da Lotofácil a partir do histórico.

O módulo combina diferentes componentes:
- leitura de planilha Excel com o histórico completo;
- construção de uma matriz booleana (concursos x dezenas);
- estatísticas de frequência e tendência (momento e recência);
- análise estrutural via matriz de coocorrência e agrupamento simples;
- sistema de *score* ponderado para priorizar dezenas;
- geração de jogos de 15 a 18 dezenas seguindo restrições básicas.

A intenção é oferecer uma base clara para calibrar *backtests* e
experimentar ajustes de pesos conforme a estratégia desejada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

NumberScore = Dict[int, float]
CoocMatrix = np.ndarray


@dataclass
class ScoreConfig:
    """Configuração de pesos do sistema de pontuação."""

    weight_freq: float = 0.35
    weight_momentum: float = 0.25
    weight_recency: float = 0.20
    weight_pairs: float = 0.20

    min_pair_support: float = 0.08  # suporte mínimo para considerar um par forte
    momentum_window: int = 30       # concursos usados para tendência recente

    def normalized_weights(self) -> Tuple[float, float, float, float]:
        total = self.weight_freq + self.weight_momentum + self.weight_recency + self.weight_pairs
        return (
            self.weight_freq / total,
            self.weight_momentum / total,
            self.weight_recency / total,
            self.weight_pairs / total,
        )


# ============================================================
#  CARREGAMENTO E MATRIZ HISTÓRICA
# ============================================================

def load_history(file_path: Path | str) -> pd.DataFrame:
    """Lê uma planilha de histórico da Lotofácil.

    A planilha deve conter pelo menos 25 colunas numeradas de 1 a 25
    (ou "Dezena 1", "D1", etc.). O carregamento é tolerante a cabeçalhos
    mais longos, desde que haja 25 colunas numéricas no início.
    """

    df_raw = pd.read_excel(file_path)
    numeric_cols = [c for c in df_raw.columns if _is_number_column(c)]
    if len(numeric_cols) < 25:
        raise ValueError("A planilha precisa ter pelo menos 25 colunas numéricas com as dezenas.")

    df = df_raw[numeric_cols[:25]].copy()
    df.columns = list(range(1, 26))
    return df


def build_boolean_matrix(df: pd.DataFrame) -> np.ndarray:
    """Retorna uma matriz (concursos x 25) com 1 para dezenas sorteadas."""

    matrix = np.zeros((len(df), 25), dtype=int)
    for idx, row in df.iterrows():
        tens = [int(x) for x in row.tolist() if not pd.isna(x)]
        matrix[idx, [t - 1 for t in tens]] = 1
    return matrix


def _is_number_column(col_name: object) -> bool:
    if isinstance(col_name, (int, float)):
        return True
    if isinstance(col_name, str):
        stripped = "".join(ch for ch in col_name if ch.isdigit())
        return bool(stripped)
    return False


# ============================================================
#  ESTATÍSTICAS BÁSICAS
# ============================================================

def frequency_scores(matrix: np.ndarray) -> NumberScore:
    counts = matrix.sum(axis=0)
    total_draws = matrix.shape[0]
    freq = counts / max(total_draws, 1)
    return {i + 1: float(freq[i]) for i in range(25)}


def momentum_scores(matrix: np.ndarray, window: int) -> NumberScore:
    recent = matrix[-window:] if matrix.shape[0] >= window else matrix
    counts = recent.sum(axis=0)
    freq = counts / max(len(recent), 1)
    return {i + 1: float(freq[i]) for i in range(25)}


def recency_scores(matrix: np.ndarray) -> NumberScore:
    last_seen = np.argmax(np.flip(matrix, axis=0), axis=0)
    distances = last_seen + 1  # concursos desde a última aparição
    max_dist = float(distances.max()) or 1.0
    norm = 1 - (distances / max_dist)  # quanto menor a distância, maior o score
    return {i + 1: float(norm[i]) for i in range(25)}


def pair_cooccurrence(matrix: np.ndarray) -> CoocMatrix:
    """Matriz 25x25 com suporte relativo de pares (diagonal = 0)."""

    total = matrix.shape[0]
    cooc = (matrix.T @ matrix) / max(total, 1)
    np.fill_diagonal(cooc, 0.0)
    return cooc


def cluster_from_pairs(cooc: CoocMatrix, threshold: float) -> List[Set[int]]:
    """Agrupa dezenas conectadas por pares fortes usando busca em grafo."""

    adjacency: Dict[int, Set[int]] = {i: set() for i in range(25)}
    for i in range(25):
        for j in range(i + 1, 25):
            if cooc[i, j] >= threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited: Set[int] = set()
    clusters: List[Set[int]] = []

    def dfs(node: int, group: Set[int]):
        visited.add(node)
        group.add(node)
        for nb in adjacency[node]:
            if nb not in visited:
                dfs(nb, group)

    for n in range(25):
        if n not in visited:
            group: Set[int] = set()
            dfs(n, group)
            clusters.append({g + 1 for g in group})
    return clusters


# ============================================================
#  SISTEMA DE SCORE
# ============================================================

def normalize_scores(scores: NumberScore) -> NumberScore:
    values = np.array(list(scores.values()), dtype=float)
    min_v, max_v = values.min(), values.max()
    if math.isclose(min_v, max_v):
        return {k: 0.5 for k in scores}
    norm = (values - min_v) / (max_v - min_v)
    return {k: float(v) for k, v in zip(scores.keys(), norm)}


def compute_score_table(matrix: np.ndarray, cfg: Optional[ScoreConfig] = None) -> NumberScore:
    cfg = cfg or ScoreConfig()
    w_freq, w_mom, w_rec, w_pair = cfg.normalized_weights()

    freq = normalize_scores(frequency_scores(matrix))
    mom = normalize_scores(momentum_scores(matrix, cfg.momentum_window))
    rec = normalize_scores(recency_scores(matrix))

    cooc = pair_cooccurrence(matrix)
    pair_strength = normalize_scores({i + 1: float(cooc[i].mean()) for i in range(25)})

    score: NumberScore = {}
    for n in range(1, 26):
        score[n] = (
            w_freq * freq[n]
            + w_mom * mom[n]
            + w_rec * rec[n]
            + w_pair * pair_strength[n]
        )
    return score


# ============================================================
#  GERAÇÃO DE JOGOS
# ============================================================

def _target_parity(size: int) -> Tuple[int, int]:
    # heurística aproximada para Lotofácil
    if size == 15:
        return 8, 7
    if size == 16:
        return 8, 8
    if size == 17:
        return 9, 8
    return 9, 9  # size == 18


def _select_seed_numbers(scores: NumberScore, clusters: List[Set[int]], size: int) -> List[int]:
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    chosen: List[int] = []

    # garantir cobertura mínima de clusters fortes
    for cluster in sorted(clusters, key=len, reverse=True):
        for n in sorted(cluster, key=lambda x: scores[x], reverse=True):
            if n not in chosen:
                chosen.append(n)
                break
        if len(chosen) >= size:
            return chosen[:size]

    for n, _ in ordered:
        if n not in chosen:
            chosen.append(n)
        if len(chosen) >= size:
            break
    return chosen[:size]


def _refine_with_pairs(
    current: List[int],
    scores: NumberScore,
    cooc: CoocMatrix,
    size: int,
    rng: np.random.Generator,
) -> List[int]:
    available = [n for n in range(1, 26) if n not in current]
    while len(current) < size and available:
        candidate = max(available, key=lambda n: scores[n])
        available.remove(candidate)

        # reforçar pares fortes
        partners = [c for c in current if cooc[c - 1, candidate - 1] > 0]
        if partners and rng.random() < 0.6:
            current.append(candidate)
        elif not partners:
            current.append(candidate)
    return current[:size]


def _adjust_parity(game: List[int], target_par: Tuple[int, int]) -> List[int]:
    even_target, odd_target = target_par
    evens = [n for n in game if n % 2 == 0]
    odds = [n for n in game if n % 2 == 1]

    def replace(from_list: List[int], to_list: List[int], need: int) -> None:
        while len(from_list) > need and to_list:
            swap_in = to_list.pop(0)
            swap_out = from_list.pop()
            game.remove(swap_out)
            if swap_in not in game:
                game.append(swap_in)

    replace(evens, odds, even_target)
    replace(odds, evens, odd_target)
    return sorted(game)


def generate_candidate_game(
    matrix: np.ndarray,
    scores: NumberScore,
    cooc: CoocMatrix,
    clusters: List[Set[int]],
    size: int,
    seed: Optional[int] = None,
) -> List[int]:
    rng = np.random.default_rng(seed)
    base = _select_seed_numbers(scores, clusters, size)
    refined = _refine_with_pairs(base, scores, cooc, size, rng)
    parity_adjusted = _adjust_parity(refined, _target_parity(size))
    return parity_adjusted


def suggest_games(
    df_history: pd.DataFrame,
    cfg: Optional[ScoreConfig] = None,
    sizes: Sequence[int] = (15, 16, 17, 18),
    seed: Optional[int] = None,
) -> Dict[int, List[int]]:
    """Gera sugestões de jogos por tamanho.

    Retorna um dicionário `{tamanho: lista_de_dezenas}`.
    """

    cfg = cfg or ScoreConfig()
    matrix = build_boolean_matrix(df_history)
    cooc = pair_cooccurrence(matrix)
    clusters = cluster_from_pairs(cooc, cfg.min_pair_support)
    scores = compute_score_table(matrix, cfg)

    return {
        size: generate_candidate_game(matrix, scores, cooc, clusters, size, seed=seed)
        for size in sizes
    }


# ============================================================
#  BACKTEST SIMPLES
# ============================================================

def hit_count(game: Iterable[int], result: Iterable[int]) -> int:
    return len(set(game) & set(result))


def backtest_last_draw(
    df_history: pd.DataFrame,
    cfg: Optional[ScoreConfig] = None,
    sizes: Sequence[int] = (15, 16, 17, 18),
    seed: Optional[int] = None,
) -> Dict[int, int]:
    """Avalia as sugestões contra o último concurso da série."""

    past = df_history.iloc[:-1]
    real_result = df_history.iloc[-1].tolist()

    suggestions = suggest_games(past, cfg=cfg, sizes=sizes, seed=seed)
    return {size: hit_count(game, real_result) for size, game in suggestions.items()}


# ============================================================
#  EXEMPLO DE USO EM LINHA DE COMANDO
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gerador de jogos da Lotofácil baseado em histórico.")
    parser.add_argument("arquivo", type=Path, help="Planilha Excel com o histórico completo.")
    parser.add_argument("--seed", type=int, default=None, help="Seed para reprodutibilidade.")
    parser.add_argument("--window", type=int, default=30, help="Janela de tendência recente.")
    parser.add_argument("--pair-support", type=float, default=0.08, help="Suporte mínimo para pares fortes.")
    args = parser.parse_args()

    cfg = ScoreConfig(momentum_window=args.window, min_pair_support=args.pair_support)
    history = load_history(args.arquivo)
    games = suggest_games(history, cfg=cfg, seed=args.seed)

    print("Sugestões de jogos (ordem crescente):")
    for size, game in games.items():
        print(f"{size} dezenas: {sorted(game)}")

    print("\nBacktest simples no último concurso:")
    hits = backtest_last_draw(history, cfg=cfg, seed=args.seed)
    for size, acertos in hits.items():
        print(f"Jogo de {size}: {acertos} acertos")
