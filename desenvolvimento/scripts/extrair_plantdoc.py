"""Extrai o .zip do PlantDoc no Windows, tratando nomes inválidos e longos.

O ``git clone`` do PlantDoc falha no Windows: parte dos nomes de arquivo foi
copiada de URLs e contém ``?``, proibido no NTFS. Este script extrai o ``.zip``
gerado pelo GitHub preservando o conteúdo das imagens e alterando só os nomes:

- remove o prefixo de primeiro nível do GitHub (``PlantDoc-Dataset-master/``),
  deixando a mesma estrutura de um clone (``train/``, ``test/``);
- troca os caracteres proibidos no NTFS por ``_``;
- encurta nomes com mais de 80 caracteres para ``stem[:60] + "_" + sha1[:8]``,
  evitando o limite de 260 caracteres de caminho;
- desambigua nomes que só diferem em maiúsculas (imagens diferentes que o NTFS
  trataria como o mesmo arquivo);
- grava ``_renomeados.csv`` com o par original → salvo de cada arquivo alterado.

Uso (a partir da raiz do projeto):
    python -I desenvolvimento/scripts/extrair_plantdoc.py plantdoc.zip desenvolvimento/dataset_dominio_bruto/nao_alface/plantdoc

Referência: docs/VERIFICADOR_DOMINIO.md, Seção 3.
"""

import csv
import hashlib
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

PROIBIDOS = re.compile(r'[<>:"\\|?*]')
LIMITE_NOME = 80


def nome_seguro(nome: str) -> str:
    """Converte um nome de arquivo em um nome válido e curto no NTFS.

    Args:
        nome: Nome original do arquivo dentro do ``.zip``.

    Returns:
        O nome sem caracteres proibidos e com no máximo ~70 caracteres quando
        o original passa de :data:`LIMITE_NOME`.
    """
    limpo = PROIBIDOS.sub("_", nome).rstrip(" .")
    if len(limpo) <= LIMITE_NOME:
        return limpo
    caminho = PurePosixPath(limpo)
    sufixo = caminho.suffix if len(caminho.suffix) <= 6 else ""
    codigo = hashlib.sha1(nome.encode("utf-8")).hexdigest()[:8]
    return f"{caminho.stem[:60]}_{codigo}{sufixo}"


def main() -> int:
    """Ponto de entrada: ``extrair_plantdoc.py <arquivo.zip> <pasta_destino>``."""
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    arquivo_zip, destino = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
    destino.mkdir(parents=True, exist_ok=False)

    renomeados: list[tuple[str, str]] = []
    gravados = 0
    with zipfile.ZipFile(arquivo_zip) as zf:
        for membro in zf.infolist():
            if membro.is_dir():
                continue
            partes = PurePosixPath(membro.filename).parts[1:]  # remove o prefixo do GitHub
            if not partes or any(p in ("..", "") for p in partes):
                print(f"IGNORADO (caminho suspeito): {membro.filename}")
                continue

            pastas = [PROIBIDOS.sub("_", p).rstrip(" .") for p in partes[:-1]]
            original = partes[-1]
            salvo = nome_seguro(original)

            alvo = destino.joinpath(*pastas, salvo).resolve()
            if alvo.exists():
                # O NTFS não distingue maiúsculas: "CAR1.jpg" e "car1.jpg" são
                # imagens diferentes no repositório, mas o mesmo arquivo aqui.
                caminho = PurePosixPath(salvo)
                codigo = hashlib.sha1(original.encode("utf-8")).hexdigest()[:8]
                salvo = f"{caminho.stem}_{codigo}{caminho.suffix}"
                alvo = destino.joinpath(*pastas, salvo).resolve()
            if destino not in alvo.parents:
                print(f"IGNORADO (fora do destino): {membro.filename}")
                continue
            if alvo.exists():
                raise SystemExit(f"Colisão de nome: {alvo}")

            alvo.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(membro) as origem, open(alvo, "wb") as saida:
                saida.write(origem.read())
            gravados += 1
            if salvo != original:
                renomeados.append(("/".join(partes), "/".join([*pastas, salvo])))

    with open(destino / "_renomeados.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["original", "salvo"])
        escritor.writerows(renomeados)

    print(f"Arquivos gravados: {gravados}")
    print(f"Arquivos renomeados: {len(renomeados)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
