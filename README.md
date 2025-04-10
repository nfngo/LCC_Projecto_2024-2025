# LCC - Projeto 2024/2025  
## Bases de Dados Vetoriais e Imagens

Este projeto tem como objetivo...

### 🔍 Descrição

A aplicação desenvolvida permite o processamento de imagens, extração de embeddings (representações vetoriais) e armazenamento dos mesmos numa base vetorial. Isso possibilita tarefas como busca por similaridade visual entre imagens.

### 🛠️ Tecnologias Utilizadas

O projeto foi desenvolvido em Python, utilizando as seguintes bibliotecas:

- [`qdrant-client`](https://github.com/qdrant/qdrant-client) – Interface para comunicação com a base de dados vetorial Qdrant
- [`transformers`](https://github.com/huggingface/transformers) – Modelos pré-treinados para extração de embeddings
- [`numpy`](https://numpy.org/) – Manipulação de dados numéricos e vetores
- [`torch`](https://pytorch.org/) – Framework de deep learning
- [`pillow`](https://python-pillow.org/) – Processamento de imagens
- [`mtcnn`](https://github.com/ipazc/mtcnn) – Detecção facial
- [`tensorflow-cpu`](https://www.tensorflow.org/) – Alternativa leve de execução de modelos com TensorFlow
- [`Docker`](https://www.docker.com/) – Containerização e execução da base de dados Qdrant

### 🐳 Inicialização da Base de Dados (Qdrant)

Para executar o Qdrant localmente via Docker, utilize o seguinte comando:

```bash
docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### 📁 Estrutura do Projeto