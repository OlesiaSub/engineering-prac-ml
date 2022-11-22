### HW2

### Шаги:

#### Установка пакетного менеджера
я с винды :'(
```
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```
и добавила вручную в path

#### Развертывание окружения
```
poetry install
```

Если dev не нужен, то
`poetry install --without dev`

#### Сборка пакета

```
poetry config repositories.test-pypi https://test.pypi.org/legacy/
poetry config pypi-token.test-pypi <PYPI-TOKEN>
poetry build
poetry publish -r test-pypi
```

#### Ссылка на пакет в pypi-test
[test.pypi.org/project/perceptron/](https://test.pypi.org/project/perceptron/)

#### Установка пакета из pypi-test
```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple perceptron
```
