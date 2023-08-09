# NLP
Natural Language Processing, 자연어 처리

## BERT
- 대용량 원시 텍스트로부터 어휘의 양방향 문맥 정보화 문장 간의 선후 관계를 학습히여 단어를 문맥을 반영한 벡터로 표현하는 모델

## KcBERT
- Korean comment BERT
- BERT 모델에 댓글과 대댓글을 수집해 구어체 특징, 신조어, 오탈자 등을 반영한 모델
- 양방향 문맥정보를 학습하는 BERT 모델을 사용해 빈 칸 채우기 가능
- [KcBERT.ipynb](src/KcBERT/KcBERT.ipynb)

## KcELECTRA
- KcBERT보다 더 많은 dataset, 더 큰 general vocabulary를 통해 더 높은 성능을 보이는 모델
- [KcELECTRA.ipynb](src/KcELECTRA/KcELECTRA.ipynb)

## KoLLaMa
- LLaMa(Large Language model Meta AI): 더 작은 모델을 더 많은 token으로 학습하는 고성능 모델, LLaMa(13B)가 GPT-3(175B)에 비해 모델 사이즈가 10%도 안되지만 성능은 LLaMa가 GPT-3를 압도
- KoLLaMa: 한국어 데이터를 학습시킨 LLaMa 모델
- [KoLLaMa.ipynb](src/KoLLaMa/KoLLaMa.ipynb)


## KoAlpaca
- 한국어 명령어를 이해하는 오픈소스 언어모델
- 대화의 맥락과 사용자의 추가 입력의 맥락을 이해하는 모델


# System 설정
- NVIDIA Geforce RTX 3060 12GB
- python 3.11.4
- CUDA toolkit 11.8.89
- PyCharm 2023.2
- PyTorch 2.0.1


# pip install
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate xformers nvidia-ml-py3
```


