import requests
from bs4 import BeautifulSoup
import pandas as pd
'''links = []
for i in range(26,2526, 25):

    response = requests.get(f'https://search.folha.uol.com.br/search?q=d%C3%B3lar&site%5B%5D=online%2Fdinheiro&periodo=todos&sr={i}&results_count=100000&search_time=0%2C041&url=https%3A%2F%2Fsearch.folha.uol.com.br%2Fsearch%3Fq%3Dd%25C3%25B3lar%26site%255B%255D%3Donline%252Fdinheiro%26periodo%3Dtodos%26sr%3D{i}')
    content = response.content
    site = BeautifulSoup(content, 'html.parser')
    #HTML da notícia
    noticias = site.findAll('div', attrs={'class': 'c-headline__content'})

    for noticia in noticias:
        titulo = noticia.find('a')
        link = titulo['href']
        links.append(link)

ll = pd.DataFrame(links)
ll.to_csv('links_dolar_folha.csv', index=None)'''

links = pd.read_csv('links_dolar_folha.csv')['links']
conteudos = []
datas = []
for link in links:
        response = requests.get(link)
        content = response.content
        site = BeautifulSoup(content, 'html.parser')
        div_body = site.find('div', attrs={'class': 'c-news__body'})
        if div_body is None:
            pass
        else:
            textos = [a.text for a in div_body.findAll('p')]
            if len(' '.join(textos)) > 10:
                conteudos.append(' '.join(textos))
                print((site.find('time').text))
                data_publicacao = site.find('time', attrs={'itemprop': 'datePublished'})
                datas.append(data_publicacao.text)

df = pd.DataFrame()
df['Data'] = datas
df['Textos'] = conteudos

#df.to_excel('teste_folha.xlsx', index=None)




