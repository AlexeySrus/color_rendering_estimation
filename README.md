# Images color rendering accuracy analysis after applying noise reduction models

## Ссылки на данные
По данной [ссылке](https://disk.yandex.ru/d/H-GxaeR01etD8w "Yandex Disk") можно скачать 
тестовый набор изображений с извлеченными цветами из цветовой шкалы.

Таблицы со значениями результатов экспериментов находятся в директории `data`.

## Запуск моделей шумоподавления
Для запуска кода шумоподалвения необходимо сначала скачать веса моделей.
Это можно сделать по следующей [ссылке]( "Yandex Disk").
Содержимое архива неоходимо разорхивировать в директорию `data`.

Для подавления шума на изображениях в директории можно воспользоваться следующим
скриптом: `denoise_images.py`
```shell
usage: Calibrate images in folder [-h] [-i INPUT] [-o OUTPUT] [-a {nlmd,danet-pp,danet,danet-t}] [--device {cuda,cpu,mps}]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to input folder with images
  -o OUTPUT, --output OUTPUT
                        Path to output folder with calibrated
  -a {nlmd,danet-pp,danet,danet-t}, --algorithm {nlmd,danet-pp,danet,danet-t}
                        Denoising approach: 'nlmd' is Non-local means denoising, 'danet' is Unet-likely model trained with
                        DANet strategy, 'danet-pp' is Unet-likely model trained with DANet+ strategy, 'danet-t' is
                        Uformer model trained with DANet strategy
  --device {cuda,cpu,mps}
                        Inference device (for Apple M1/M2 chip use 'mps')
```