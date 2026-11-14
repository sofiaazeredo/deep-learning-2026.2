Nada de hiperparâmetro fixo no código. configs/defaults/ são pontos de partida marcados como não validados. O src/sweep.py faz busca aleatória só no val e grava a escolha em configs/selected/.
Os scripts recusam rodar sem essa escolha, a menos que você passe --allow_defaults. Testei a recusa.
O backbone, o tamanho de entrada (data.size) e o modo de redimensionamento (resize_mode, squash ou pad) também entram no sweep.
Schema dos CSVs detectado, não chutado. tests/check_data confere o pacote contra o enunciado (contagens, orçamentos aninhados, splits disjuntos) e grava as colunas detectadas. Se achar o nome ambíguo, falha, e você força com --image_col e --label_col. Testei rótulo por nome, rótulo numérico e image_id com extensão.
Guarda do test_public. Os sweeps nem carregam o test_public, e há um teste que verifica isso. Cada run avalia o teste uma vez.
Arquivos que eram placeholder: predict.py, saliencia.py, galeria.py e grafico_orcamento.py agora existem, mais select_final.py, avaliar.py, README.md, AI_LOG.md, requirements.txt, .gitignore e scripts/reproduzir.sh.
Mudança de design no FixMatch: a vista forte agora parte do mesmo recorte da fraca. Com animal de 2% do quadro, recortes independentes poderiam deixar o animal fora de uma das vistas.
O pandas novo não usa mais dtype == object para texto, o que quebraria a leitura do classes.csv real.
cm.get_cmap não existe mais no matplotlib novo.
O hook do Grad-CAM quebrava durante a oclusão.
O ids numéricos do gt no avaliar seriam comparados com nomes de classe.
