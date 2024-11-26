import requests

def test_api_check_image():
    url = "http://0.0.0.0:8000/api/checkImage"
    file_path = "/home/aazamatov/Downloads/onnx-lp-recognizer-api/application/test/test_images/photo_2022-11-07_14-21-03.jpg"

    # Загрузить файл в запрос
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(url, files=files)

    # Проверить результат
    print(response.status_code)
    print(response.json())

if __name__ == "__main__":
    test_api_check_image()
