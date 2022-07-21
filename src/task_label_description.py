korean_singular_conversation_labels = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']

# Description 1
korean_singular_conversation_label_descriptions = {
    key: key + '의 감정을 표현한 것'
    for key in korean_singular_conversation_labels
}

# Description 2
# korean_singular_conversation_label_descriptions = {
#     key: f'이 대화는 {key}의 감정을 표현한다'
#     for key in korean_singular_conversation_labels
# }

# Description 3
# korean_singular_conversation_label_descriptions = {
#     key: f'이 말은 {key}의 감정 표현이다.'
#     for key in korean_singular_conversation_labels
# }

# Description 1
nsmc_label_descriptions = {
    '긍정': '이것은 훌륭한 영화입니다.',
    '부정': '이것은 최악의 영화입니다.',
}

TASK_LABELS_DESC = {
    "ksc": korean_singular_conversation_label_descriptions,
    "nsmc": nsmc_label_descriptions,
}
