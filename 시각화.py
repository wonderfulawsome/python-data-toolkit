########### 막대그래프 생성
plt.figure(figsize=(14, 8))

# 색상 팔레트 설정
colors = plt.cm.Set3(range(len(dong_counts)))

# 막대그래프 생성
bars = plt.bar(dong_counts.index, dong_counts.values, 
               color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

# 막대 위에 값 표시
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
# 제목과 라벨 설정
plt.title('동별 토양오염 관리대상 시설 밀집도', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('동', fontsize=12, fontweight='bold')
plt.ylabel('시설 개수', fontsize=12, fontweight='bold')

# 축 스타일링
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')

# 배경색 설정
plt.gca().set_facecolor('#f8f9fa')

# 여백 조정
plt.tight_layout()
plt.show()

