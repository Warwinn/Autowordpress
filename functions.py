import os
import requests
import random
from PIL import Image 
from wordpress_xmlrpc.compat import xmlrpc_client
from wordpress_xmlrpc.methods import media
from youtubesearchpython import VideosSearch


def _build_wp_el(el, text):
    """
        format headers and paragraph into wordpress tags

        el (str)   : html tag (i.e.: "h1", "h2", "p")
        text (str) : text to format

        return (str)
    """
    wp_type = el == "p" and "paragraph" or "heading"
    wp_elem = f"""
    <!-- wp:{wp_type} -->
    <{el}>{text}</{el}>
    <!-- /wp:{wp_type} -->
    """
    return wp_elem


def generate_wp(data,client):
    """
        generate formatted text from json,
        with wordpress tags

        data (json)

        return (str)
    """
    keys = data.keys()
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    for keyword in keys:
        for rel_key in data[keyword]:
            i = 0
            images = createimagelist(rel_key)
            file_name = '_'.join(rel_key.split(' '))
            for question in data[keyword][rel_key]:
                wp_gen = ""
                wp_gen += f"{question} - {rel_key}"
                for articles in data[keyword][rel_key][question]:
                    for article in articles:
                        subtitle = article["subtitle"].capitalize()
                        wp_gen += _build_wp_el("h2", subtitle)
                        wp_gen += _build_wp_el("p", article["paragraph"])
                        rand = random.randint(0,1)
                        if rand == 1:
                            try:
                                image = images.pop()
                            except IndexError: 
                                images = createimagelist(rel_key)
                                image = images.pop()
                            wp_gen += add_image(image,client)
                videosSearch = VideosSearch(str(question) + ' france', limit = 5, region = 'FR').result()
                youtube_vid = videosSearch['result'][random.randrange(5)]['id']
                wp_gen += _build_wp_el("h2", "Related Vidéo")
                print(youtube_vid)
                wp_gen += f'<iframe src="https://www.youtube.com/embed/{youtube_vid}"width="560" height="315" frameborder="0" allowfullscreen="allowfullscreen"></iframe>'



                with open(f"outputs/{file_name}{i}.txt", 'w') as outfile:
                    outfile.write(wp_gen)
                print(f"{file_name}{i}.txt created")
                i += 1
def createimagelist(keyword):
    url = f'https://unsplash.com/napi/search?query={keyword}&xp=&per_page=50' 
    headers = {'authorization': 'Client-ID oaxc1_ApaTf2OdpVDTK-4FlHedmzgJq_qaIo0OYyfEA'}
    response = requests.get(url, headers=headers)

    list_response_img=[]
    for i in range (len(response.json()["photos"]["results"])):
        list_response_img.append (response.json()["photos"]["results"][i]['urls']['small'])
    random.shuffle(list_response_img)   
    return list_response_img

def add_image(image,client):

    img_data = requests.get(image).content
    data = {'name': 'picture.jpg', 'type': 'image/jpeg'}
    

    with open('image_name.jpg', 'wb') as handler:
        output=[]   
        handler.write(img_data)

        picture = Image.open('image_name.jpg')

        picture.save('Compressed_imagename'+ str(0) + '.jpg', optimize=True, quality=50)

        with open('Compressed_imagename'+ str(0) + '.jpg', 'rb') as img:
            data['bits'] = xmlrpc_client.Binary(img.read())

        response_img = client.call(media.UploadFile(data))

        output.append(response_img['url'])

    return '<figure class="wp-block-image size-large"><img src='+ output[0] +' alt="" class="wp-image-113"/></figure>'



