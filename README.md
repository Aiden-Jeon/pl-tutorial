# pl-tutorial
pytorch ligthning tutorial

## 시작하기

make 명령어가 사용되는 환경에서만 사용할 수 있습니다.
`make init` 명령어를 통해 패키지를 설치합니다.

설치 이후 다음과 같이 `boostcamp` 패키지를 이용할 수 있습니다.

```python
import boostcamp
```

## requirements.txt

새로운 패키지를 설치할 경우 아래 명령어를 통해서 `requirements.txt`를 수정해야 합니다.

```bash
pip freeze --exclude boostcamp
```
