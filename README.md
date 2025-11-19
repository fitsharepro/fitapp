# fitapp

Ferramentas em Python para explorar o histórico da Lotofácil e sugerir novas apostas
usando matrizes, clusters de coocorrência e um sistema de pontuação calibrável.

## Visão geral
- Leitura direta de planilha Excel com todos os concursos já sorteados.
- Conversão para uma matriz booleana (concursos × 25 dezenas) para facilitar análises.
- Cálculo de frequência, recência e tendência (janela móvel) para cada dezena.
- Mapeamento estrutural com matriz de coocorrência, agrupando dezenas correlacionadas.
- Sistema de score ponderado que combina frequência, momento, recência e força dos pares.
- Geração de sugestões de jogos de 15, 16, 17 e 18 dezenas com paridade balanceada.
- Backtest simples comparando as sugestões com o último concurso do histórico.

## Requisitos
- Python 3.10+
- Dependências: `pandas` e `numpy` (`pip install pandas numpy`).

## Uso rápido
1. Salve a planilha histórica (ex.: `lotofacil_historico.xlsx`) na raiz do projeto.
2. Execute o gerador via linha de comando:

```bash
python lotofacil_generator.py lotofacil_historico.xlsx --seed 42 --window 30 --pair-support 0.08
```

O script imprimirá sugestões de jogos para 15, 16, 17 e 18 dezenas, além de um
backtest simples contra o último concurso disponível.

## Ajustando a estratégia
- **window** (`--window`): número de concursos usados para medir o momento recente.
- **pair-support** (`--pair-support`): suporte mínimo para considerar um par forte e formar clusters.
- **ScoreConfig**: edite os pesos em `lotofacil_generator.py` para dar mais ênfase a frequência,
  recência, momento ou pares.
