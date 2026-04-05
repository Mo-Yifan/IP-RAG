// src/api/static/js/main.js

// 等待 DOM 完全加载
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 IP-RAG Frontend Initialized'); // 调试日志

    // 🔴 1. 获取 DOM 元素
    const questionInput = document.getElementById('question');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.getElementById('btn-text');
    const btnLoading = document.getElementById('btn-loading');
    const resultArea = document.getElementById('result-area');
    const errorArea = document.getElementById('error-area');
    const patentList = document.getElementById('patent-list');
    const answerDiv = document.getElementById('answer');
    
    // ✅ 修复：获取错误信息元素 (之前漏掉了这个)
    const errorMessage = document.getElementById('error-message');

    // 安全检查
    if (!questionInput || !submitBtn || !resultArea) {
        console.error('❌ IP-RAG 页面初始化失败：未找到必要的 DOM 元素');
        return;
    }

    // 🔴 2. 绑定点击事件
    submitBtn.addEventListener('click', async () => {
        const question = questionInput.value.trim();
        
        // 输入验证
        if (!question) {
            showError("请输入您的技术问题！");
            return;
        }

        console.log('🔍 开始检索:', question); // 调试日志
        setLoadingState(true);

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    top_k: 3,
                    valid_only: document.getElementById('active_only')?.checked || false
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }

            const data = await response.json();
            console.log('✅ 检索成功:', data); // 调试日志
            renderSuccess(data);
            
        } catch (error) {
            console.error('❌ 检索请求失败:', error);
            showError(`检索失败: ${error.message}`);
        } finally {
            setLoadingState(false);
        }
    });

    // 🔴 3. 支持回车键提交
    questionInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // 防止换行
            submitBtn.click();  // 触发点击事件
        }
    });

    // =================== //
    // 🎨 UI 状态管理函数 //
    // =================== //

    function setLoadingState(loading) {
        submitBtn.disabled = loading;
        if (btnText) btnText.textContent = loading ? "🧠 深度思考中..." : "🔍 开始检索";
        if (btnLoading) btnLoading.classList.toggle('hidden', !loading);
    }

    function showError(message) {
        // 隐藏可能存在的旧结果
        resultArea.classList.add('hidden');
        
        // ✅ 修复：使用已定义的 errorMessage 变量
        if (errorMessage) {
            errorMessage.textContent = message;
        }
        errorArea.classList.remove('hidden');
        
        // 恢复按钮
        setLoadingState(false);
    }

    function renderSuccess(data) {
        // 1. 渲染 AI 摘要
        if (answerDiv) {
            answerDiv.textContent = data.answer || "未生成具体摘要。";
        }

        // 2. 渲染专利列表
        if (patentList) {
            patentList.innerHTML = ''; // 清空旧列表
            
            if (data.citations && data.citations.length > 0) {
                data.citations.forEach((patent, index) => {
                    const patentCard = document.createElement('div');
                    // 使用 Tailwind 类名以保持风格一致
                    patentCard.className = "bg-gray-50 p-6 rounded-xl hover:bg-white transition border border-gray-100 shadow-sm mb-4";
                    
                    patentCard.innerHTML = `
                        <div class="flex flex-col md:flex-row md:justify-between md:items-start gap-4">
                            <div class="flex-1">
                                <div class="flex items-center gap-3 mb-2">
                                    <span class="font-bold text-2xl text-cyan-600">[${index + 1}]</span>
                                    <h3 class="font-bold text-xl text-gray-800">${patent.title || '无标题'}</h3>
                                </div>
                                <div class="text-sm text-gray-500 mb-3">
                                    <span class="font-mono bg-gray-200 px-2 py-1 rounded">ID: ${patent.patent_id || 'N/A'}</span>
                                    <span class="ml-4">相似度: ${(patent.score || 0).toFixed(3)}</span>
                                </div>
                                <p class="text-gray-600 text-base leading-relaxed">
                                    <strong>摘要:</strong> ${patent.abstract_snippet || '无摘要信息'}
                                </p>
                            </div>
                            <div class="mt-4 md:mt-0">
                                <a href="https://patents.google.com/patent/${patent.patent_id}" target="_blank" 
                                   class="inline-flex items-center px-4 py-2 bg-cyan-600 text-white text-sm font-semibold rounded hover:bg-cyan-700 transition">
                                    查看全文 &rarr;
                                </a>
                            </div>
                        </div>
                    `;
                    
                    patentList.appendChild(patentCard);
                });
            } else {
                patentList.innerHTML = '<p class="text-gray-500 text-center py-4">未找到相关专利。</p>';
            }
        }

        // 3. 切换显示状态
        errorArea.classList.add('hidden');
        resultArea.classList.remove('hidden');
        
        // 自动滚动到底部查看结果
        resultArea.scrollIntoView({ behavior: 'smooth' });
    }
});