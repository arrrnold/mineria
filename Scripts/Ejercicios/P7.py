# ejemplo de SaaS
import requests
import json

def get_github_repo_info(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(f"Nombre del repo: {data['name']}")
        print(f"Descripcion: {data['description']}")
        print(f"Estrellas: {data['stargazers_count']}")
        print(f"Idioma principal: {data['language']}")
    else:
        print(f"NO se puede obtener informacion para el repo {owner}/{repo}. Estado HTTP: {response.status_code}")

get_github_repo_info("arrrnold","reporte-alergias")