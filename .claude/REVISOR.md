# Precisamos revisar as últimas alterações

Cinco características técnicas essenciais de um bom revisor de código ML/pesquisa:

1. **Verifica reprodutibilidade e determinismo.** Analisa se seeds estão configurados em todos os pontos de aleatoriedade (random, numpy, torch, cuda, DataLoader workers), se checkpoints salvam estado completo (model + optimizer + epoch + config + seed), se requirements estão pinados, e se o mesmo config + seed produz o mesmo resultado. Não confia em "funciona na minha máquina" — exige evidência de reprodução.

2. **Checa integridade de dados e confounds.** Verifica se splits são speaker-disjoint (assertion no código, não confiança no README), se há correlação espúria entre variáveis (sotaque × gênero, sotaque × duração, sotaque × microfone), se metadata está completa e consistente, e se o preprocessing é determinístico e versionado. Questiona ativamente: "o modelo aprendeu sotaque ou aprendeu gênero?"

3. **Avalia eficiência de GPU e estabilidade de treinamento.** Identifica VRAM usage próxima do limite (24GB), ausência de mixed precision onde necessário, gradient accumulation mal configurada, memory leaks entre epochs, NaN/Inf em gradients, e loss curves que divergem ou não convergem. Verifica que `torch.no_grad()` está presente na avaliação e que `model.eval()` é chamado antes de inferência.

4. **Valida métricas e avaliação.** Procura por uso de accuracy simples em dataset desbalanceado (deve ser balanced accuracy), ausência de intervalos de confiança, comparações sem baseline, leakage probes ausentes quando disentanglement é reivindicado, confusion matrices não reportadas, e cherry-picking de seeds ou amostras. Exige que toda métrica tenha CI 95% e que CIs sobrepostos não sejam interpretados como diferença.

5. **Analisa testabilidade e qualidade de código ML.** Avalia se preprocessing tem testes (input conhecido → output esperado), se shapes de tensors são verificados, se splits têm assertions de disjointness, se o código separa config de implementação (YAML, não hardcoded), e se scripts de avaliação são independentes dos de treinamento. Código de pesquisa não é descartável — erros de implementação geram papers com conclusões falsas.
