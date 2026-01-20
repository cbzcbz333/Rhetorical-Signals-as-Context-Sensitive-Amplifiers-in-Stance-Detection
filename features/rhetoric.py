# features/rhetoric.py
import re
from typing import List, Dict, Any
from feature_extractor import OptimizedFeatureExtractor
import requests
import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import hashlib
from contrast_extractor import extract_contrast_features

def simple_length_norm(text: str) -> float:
    # 你可以换成 tokenizer 长度，这里用字符长度做最简单归一化
    return max(1.0, len(text))

class RhetoricFeatureEncoder:
    """
    把你现有 LLM 抽取结果 -> 定长 rf 向量
    """
    def __init__(self, api_key: str = None, cache_enabled: bool = True):
        self.ext = OptimizedFeatureExtractor(api_key=api_key, cache_enabled=cache_enabled)

    def extract_vector(self, text: str) -> List[float]:
        L = simple_length_norm(text)

        # 1) RQ
        rq = self.ext.extract_features(text, feature_type="rhetorical", use_cache=True)
        rq_q = rq.question_count
        rq_r = rq.rhetorical_count
        rq_ratio = rq.rhetorical_ratio

        # 2) MOD
        mod = self.ext.extract_features(text, feature_type="modality", use_cache=True)
        modal = float(mod.get("modal_verb_count", 0))
        hedge = float(mod.get("hedge_marker_count", 0))
        strong = float(mod.get("strong_assertion_count", 0))
        epi = float(mod.get("epistemic_strength_score", 0.0))

        # 3) CON (如果 contrast_extractor 有)
        # 这里给一个很保守的写法：拿不到就置0
        try:
            from contrast_extractor import extract_contrast_features
            con = extract_contrast_features(text)
            # 你需要把它映射成一个数：比如 con["contrast_count"]/L 或 con["has_contrast"]
            con_score = float(con.get("contrast_score", con.get("has_contrast", 0.0)))
        except Exception:
            con_score = 0.0

        # 归一化（计数/L）
        rf = [
            rq_q / L,
            rq_r / L,
            rq_ratio,
            modal / L,
            hedge / L,
            strong / L,
            epi,
            con_score
        ]
        return rf

    @property
    def rf_dim(self) -> int:
        return 8

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """特征提取结果"""
    text: str
    question_count: int
    rhetorical_count: int
    rhetorical_ratio: float
    questions: List[Dict]
    metadata: Dict
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "text": self.text,
            "features": {
                "question_count": self.question_count,
                "rhetorical_question_count": self.rhetorical_count,
                "rhetorical_question_ratio": self.rhetorical_ratio
            },
            "analysis": {
                "questions": self.questions,
                "raw_response": self.raw_response
            },
            "metadata": self.metadata
        }

    def summary(self) -> str:
        """结果摘要"""
        return (f"文本: {self.text[:50]}...\n"
                f"疑问句: {self.question_count} | "
                f"反问句: {self.rhetorical_count} | "
                f"比例: {self.rhetorical_ratio:.2f} | "
                f"模式: {self.metadata.get('method', 'unknown')}")


class OptimizedFeatureExtractor:
    """优化版特征提取器 - 专门用于立场检测"""

    def __init__(self, api_key: str = None, cache_enabled: bool = True):
        self.api_key = api_key or "sk-JhzsitLNi4ztobLxgmbdIBCPXtUPTFwFmkYdAsOILqW1xDEy"
        self.endpoint = "https://api.shubiaobiao.com/v1/chat/completions"

        # 已确认可用的模型（按优先级排序）
        self.models = [
            "gpt-4o-mini",  # 成本低，速度快
            "gpt-3.5-turbo",  # 最稳定
            "gpt-4o",  # 能力强
            "gpt-4.1-mini",  # 较新版本
            "gpt-3.5-turbo-0125",  # 特定版本
        ]

        # 缓存系统
        self.cache_enabled = cache_enabled
        self.response_cache: Dict[str, Dict] = {}
        self.cache_ttl = 3600  # 1小时

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "models_used": {}
        }

        logger.info(f"✅ 特征提取器初始化完成，可用模型: {self.models}")

    def _get_cache_key(self, text: str, feature_type: str = "rhetorical") -> str:
        """生成缓存键"""
        content = f"{feature_type}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def _save_to_cache(self, key: str, data: Dict):
        """保存到缓存"""
        if self.cache_enabled:
            self.response_cache[key] = {
                "data": data,
                "timestamp": time.time()
            }

    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """从缓存获取"""
        if not self.cache_enabled:
            return None

        if key in self.response_cache:
            entry = self.response_cache[key]
            if time.time() - entry["timestamp"] < self.cache_ttl:
                self.stats["cache_hits"] += 1
                return entry["data"]

        return None

    def _build_system_prompt(self, feature_type: str = "rhetorical") -> str:
        """构建系统提示词（按修辞特征类型）"""

        prompts = {

            # =========================
            # 第 1 类：反问 / 设问
            # =========================
            "rhetorical": """You are a linguistic analysis assistant.
    Your task is to identify rhetorical questions in short social media text.

    Definition:
    A rhetorical question is an interrogative form that does NOT seek information,
    but is used to implicitly express emphasis, criticism, or evaluation.

    Instructions:
    1. Identify all complete question sentences in the text.
    2. For each question, determine whether it is rhetorical.
    3. Do NOT infer stance polarity or sentiment.
    4. Focus only on the rhetorical function of the question.

    Criteria for rhetorical questions (any one is sufficient):
    - The question implies an obvious or assumed answer.
    - The question is used for emphasis rather than information seeking.
    - Typical rhetorical markers are present.

    Examples of rhetorical markers:
    - Chinese: 难道, 岂, 何必, 不是…吗, 怎能, 怎可
    - English: isn't it, don't you, how can, why would

    Output:
    Return ONLY a valid JSON object in the following format:

    {
      "question_count": <int>,
      "rhetorical_question_count": <int>,
      "questions": [
        {
          "text": "<question sentence>",
          "is_rhetorical": true/false
        }
      ]
    }
    """,

            # =========================
            # 第 2 类：模态 / 模糊表达
            # =========================
            "modality": """You are a linguistic analysis assistant.
    Your task is to identify modality and hedging in short social media text.

    Definition:
    Modality and hedging reflect the degree of certainty, obligation, or commitment
    expressed by the speaker, rather than sentiment or stance direction.

    Instructions:
    1. Identify modal verbs indicating obligation or possibility.
    2. Identify hedging expressions that soften or qualify claims.
    3. Identify strong assertion markers indicating high certainty.
    4. Do NOT infer stance polarity or sentiment.

    Examples:
    - Modal verbs: should, could, might, may, must
    - Hedging expressions: maybe, perhaps, it seems, I think, likely
    - Strong assertions: must, definitely, obviously, no doubt

    Output:
    Return ONLY a valid JSON object in the following format:

    {
      "modal_verb_count": <int>,
      "hedge_marker_count": <int>,
      "strong_assertion_count": <int>
    }
    """
        }

        return prompts.get(feature_type, prompts["rhetorical"])

    def _call_api_with_retry(self, messages: List[Dict], max_retries: int = 3) -> Tuple[bool, Any]:
        """带重试的API调用"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for model in self.models:
            for attempt in range(max_retries):
                try:
                    self.stats["total_requests"] += 1

                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": 1000,
                        "response_format": {"type": "json_object"}
                    }

                    logger.debug(f"尝试模型: {model} (尝试 {attempt + 1}/{max_retries})")

                    start_time = time.time()
                    response = requests.post(
                        self.endpoint,
                        headers=headers,
                        json=payload,
                        timeout=15
                    )
                    response_time = time.time() - start_time

                    if response.status_code == 200:
                        data = response.json()

                        # 记录使用统计
                        if model not in self.stats["models_used"]:
                            self.stats["models_used"][model] = 0
                        self.stats["models_used"][model] += 1

                        # 记录token使用（如果有）
                        if "usage" in data:
                            self.stats["total_tokens"] += data["usage"].get("total_tokens", 0)

                        self.stats["successful_requests"] += 1
                        logger.debug(f"✅ {model} 成功 ({response_time:.2f}s)")

                        return True, {
                            "data": data,
                            "model": model,
                            "response_time": response_time
                        }

                    else:
                        error_msg = response.text[:100] if response.text else ""
                        logger.warning(f"❌ {model} 失败: {response.status_code} - {error_msg}")

                        # 如果是临时错误，重试
                        if response.status_code in [429, 500, 502, 503, 504]:
                            wait_time = (attempt + 1) * 2  # 指数退避
                            logger.info(f"等待 {wait_time}s 后重试...")
                            time.sleep(wait_time)
                            continue
                        else:
                            # 永久错误，尝试下一个模型
                            break

                except requests.exceptions.Timeout:
                    logger.warning(f"⏱️  {model} 超时")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                except Exception as e:
                    logger.error(f"💥 {model} 异常: {e}")
                    break

        self.stats["failed_requests"] += 1
        return False, {"error": "所有模型都失败"}

    def extract_features(
            self,
            text: str,
            feature_type: str = "rhetorical",
            use_cache: bool = True
    ):
        """
        修辞特征提取主入口（支持多类修辞）

        Args:
            text: 输入文本
            feature_type: 修辞类型（rhetorical / modality / ...）
            use_cache: 是否使用缓存

        Returns:
            - rhetorical: ExtractionResult
            - modality: dict
        """

        # =========================
        # 第 1 类：反问 / 设问
        # =========================
        if feature_type == "rhetorical":

            # ---- 缓存 ----
            if use_cache and self.cache_enabled:
                cache_key = self._get_cache_key(text, "rhetorical")
                cached = self._get_from_cache(cache_key)
                if cached:
                    return ExtractionResult(
                        text=cached["text"],
                        question_count=cached["question_count"],
                        rhetorical_count=cached["rhetorical_question_count"],
                        rhetorical_ratio=cached["rhetorical_question_ratio"],
                        questions=cached.get("questions", []),
                        metadata=cached.get("metadata", {}),
                        raw_response=cached.get("raw_response")
                    )

            # ---- 构建消息 ----
            messages = [
                {"role": "system", "content": self._build_system_prompt("rhetorical")},
                {"role": "user", "content": f'Text:\n"{text}"'}
            ]

            # ---- 调用 API ----
            success, result = self._call_api_with_retry(messages)

            if success:
                try:
                    data = result["data"]
                    content = data["choices"][0]["message"]["content"]

                    features = json.loads(content)
                    features = self._validate_features(features, text)

                    extraction_result = ExtractionResult(
                        text=text,
                        question_count=features.get("question_count", 0),
                        rhetorical_count=features.get("rhetorical_question_count", 0),
                        rhetorical_ratio=features.get("rhetorical_question_ratio", 0.0),
                        questions=features.get("questions", []),
                        metadata={
                            "model": result["model"],
                            "method": "api",
                            "response_time": result["response_time"],
                            "success": True,
                            "timestamp": datetime.now().isoformat(),
                            "feature_type": "rhetorical"
                        },
                        raw_response=content[:500] + "..." if len(content) > 500 else content
                    )

                    if use_cache and self.cache_enabled:
                        cache_key = self._get_cache_key(text, "rhetorical")
                        self._save_to_cache(cache_key, extraction_result.to_dict())

                    return extraction_result

                except Exception as e:
                    logger.error(f"反问特征处理失败: {e}")
                    return self._create_fallback_result(text, str(e))

            else:
                logger.warning("反问 API 失败，使用规则匹配")
                return self._rule_based_extraction(text)

        # =========================
        # 第 2 类：模态 / 模糊表达
        # =========================
        elif feature_type == "modality":

            # ---- 缓存（独立 key，不污染第 1 类）----
            if use_cache and self.cache_enabled:
                cache_key = self._get_cache_key(text, "modality")
                cached = self._get_from_cache(cache_key)
                if cached:
                    return cached

            messages = [
                {"role": "system", "content": self._build_system_prompt("modality")},
                {"role": "user", "content": f'Text:\n"{text}"'}
            ]

            success, result = self._call_api_with_retry(messages)

            if success:
                try:
                    content = result["data"]["choices"][0]["message"]["content"]
                    data = json.loads(content)

                    modal = int(data.get("modal_verb_count", 0))
                    hedge = int(data.get("hedge_marker_count", 0))
                    strong = int(data.get("strong_assertion_count", 0))

                    denom = modal + hedge + strong
                    strength = (strong - hedge) / denom if denom > 0 else 0.0

                    features = {
                        "modal_verb_count": modal,
                        "hedge_marker_count": hedge,
                        "strong_assertion_count": strong,
                        "epistemic_strength_score": round(strength, 4)
                    }

                    if use_cache and self.cache_enabled:
                        cache_key = self._get_cache_key(text, "modality")
                        self._save_to_cache(cache_key, features)

                    return features

                except Exception as e:
                    logger.error(f"模态特征处理失败: {e}")
                    return {
                        "modal_verb_count": 0,
                        "hedge_marker_count": 0,
                        "strong_assertion_count": 0,
                        "epistemic_strength_score": 0.0
                    }

            else:
                logger.warning("模态 API 失败，使用空特征")
                return {
                    "modal_verb_count": 0,
                    "hedge_marker_count": 0,
                    "strong_assertion_count": 0,
                    "epistemic_strength_score": 0.0
                }

        # =========================
        # 未知类型（防御）
        # =========================
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    def _validate_features(self, features: Dict, original_text: str) -> Dict:
        """验证和补全特征"""
        # 基本验证
        if not isinstance(features, dict):
            features = {}

        # 确保必需字段
        features.setdefault("question_count", 0)
        features.setdefault("rhetorical_question_count", 0)
        features.setdefault("questions", [])

        # 计算比例
        qc = features["question_count"]
        rc = features["rhetorical_question_count"]
        features["rhetorical_question_ratio"] = rc / qc if qc > 0 else 0.0

        # 验证questions数组
        if not isinstance(features["questions"], list):
            features["questions"] = []

        # 清理每个问题条目
        valid_questions = []
        for q in features["questions"]:
            if isinstance(q, dict):
                # 确保必需字段
                q.setdefault("text", "")
                q.setdefault("is_rhetorical", False)


                # 确保置信度在0-1之间
                '''
                                if not isinstance(q["confidence"], (int, float)):
                    q["confidence"] = 0.5
                q["confidence"] = max(0.0, min(1.0, float(q["confidence"])))
                '''


                valid_questions.append(q)

        features["questions"] = valid_questions

        # 如果questions数量与统计不一致，修正统计
        actual_qc = len(features["questions"])
        if actual_qc != features["question_count"]:
            logger.debug(f"修正question_count: {features['question_count']} -> {actual_qc}")
            features["question_count"] = actual_qc

        actual_rc = sum(1 for q in features["questions"] if q["is_rhetorical"] is True)

        if actual_rc != features["rhetorical_question_count"]:
            logger.debug(f"修正rhetorical_count: {features['rhetorical_question_count']} -> {actual_rc}")
            features["rhetorical_question_count"] = actual_rc

        # 重新计算比例
        qc = features["question_count"]
        rc = features["rhetorical_question_count"]
        features["rhetorical_question_ratio"] = rc / qc if qc > 0 else 0.0

        return features

    def _rule_based_extraction(self, text: str) -> ExtractionResult:
        """基于规则的降级特征提取"""
        logger.info(f"使用规则匹配: {text[:50]}...")

        # 优化的反问句模式
        rhetorical_patterns = [
            # 中文模式
            (r'难道[^？?]*[？?]', "含有'难道'"),
            (r'岂[^？?]*[？?]', "含有'岂'"),
            (r'何尝[^？?]*[？?]', "含有'何尝'"),
            (r'岂不是[^？?]*[？?]', "含有'岂不是'"),
            (r'怎么(?:能|可以|可能|敢|会)[^？?]*[？?]', "含有'怎么...'结构"),
            (r'怎能[^？?]*[？?]', "含有'怎能'"),
            (r'怎可[^？?]*[？?]', "含有'怎可'"),
            (r'何必[^？?]*[？?]', "含有'何必'"),
            (r'不是[^？?]*吗[？?]', "'不是...吗'结构"),
            (r'还不[^？?]*吗[？?]', "'还不...吗'结构"),
            (r'没有[^？?]*吗[？?]', "'没有...吗'结构"),

            # 英文模式（修复问题）
            (r'(?:isn\'t|aren\'t|don\'t|doesn\'t|won\'t|can\'t)\s+it\b.*[？?]', "英文反问: isn't it结构"),
            (r'(?:isn\'t|aren\'t|don\'t|doesn\'t|won\'t|can\'t)\s+.*\b\?', "英文反问: 否定疑问"),
            (r'how\s+(?:can|could|dare|would)\s+.*[？?]', "英文反问: how can结构"),
            (r'why\s+(?:would|should)\s+.*[？?]', "英文反问: why would结构"),
            (r'what\s+(?:is|are)\s+the\s+point.*[？?]', "英文反问: what's the point"),
            (r'who\s+(?:would|could)\s+.*[？?]', "英文反问: who would"),
        ]

        # 分割句子
        sentences = re.split(r'[。！!；;\n]', text)

        questions = []
        question_count = 0
        rhetorical_count = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 检查是否是疑问句
            if sentence.endswith('？') or sentence.endswith('?'):
                question_count += 1

                is_rhetorical = False
                reason = "普通疑问句"
                #confidence = 0.3

                # 检查反问句模式
                for pattern, pattern_reason in rhetorical_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        is_rhetorical = True
                        reason = f"匹配反问模式: {pattern_reason}"
                        #confidence = 0.85
                        rhetorical_count += 1
                        break

                # 确定位置
                if question_count == 1:
                    position = "开头"
                elif question_count == len([s for s in sentences if s.endswith('？') or s.endswith('?')]):
                    position = "结尾"
                else:
                    position = "中间"

                questions.append({
                    "text": sentence,
                    "is_rhetorical": is_rhetorical
                })

        # 创建结果
        rhetorical_ratio = rhetorical_count / question_count if question_count > 0 else 0.0

        return ExtractionResult(
            text=text,
            question_count=question_count,
            rhetorical_count=rhetorical_count,
            rhetorical_ratio=rhetorical_ratio,
            questions=questions,
            metadata={
                "model": "rule_based",
                "method": "rule_based",
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "feature_type": "rhetorical",
                "note": "基于规则的降级分析"
            }
        )

    def _create_fallback_result(self, text: str, error: str) -> ExtractionResult:
        """创建降级结果"""
        return ExtractionResult(
            text=text,
            question_count=0,
            rhetorical_count=0,
            rhetorical_ratio=0.0,
            questions=[],
            metadata={
                "model": "error",
                "method": "error",
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "error": error,
                "feature_type": "rhetorical"
            }
        )

    def extract_multiple_features(self, text: str, feature_types: List[str] = None) -> Dict[str, Any]:
        """提取多种特征"""
        if feature_types is None:
            feature_types = ["rhetorical"]

        results = {}
        for feature_type in feature_types:
            if feature_type == "rhetorical":
                results[feature_type] = self.extract_features(text).to_dict()
            # 可以添加其他特征类型

        return results

    def batch_extract(self, texts: List[str], batch_size: int = 10) -> List[ExtractionResult]:
        """批量提取特征"""
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"处理批次 {i // batch_size + 1}: {len(batch)} 个文本")

            for text in batch:
                result = self.extract_features(text)
                results.append(result)

            # 批次间延迟
            if i + batch_size < len(texts):
                time.sleep(1)

        return results

    def get_stats(self) -> Dict:
        """获取统计信息"""
        success_rate = (self.stats["successful_requests"] / self.stats["total_requests"]
                        if self.stats["total_requests"] > 0 else 0)

        return {
            **self.stats,
            "success_rate": success_rate,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.response_cache),
            "current_time": datetime.now().isoformat()
        }

def _extract_modality_features(self, text: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": self._build_system_prompt("modality")
        },
        {
            "role": "user",
            "content": f'Text:\n"{text}"'
        }
    ]

    success, result = self._call_api_with_retry(messages)

    if not success:
        return {
            "modal_verb_count": 0,
            "hedge_marker_count": 0,
            "strong_assertion_count": 0,
            "epistemic_strength_score": 0.0
        }

    content = result["data"]["choices"][0]["message"]["content"]
    data = json.loads(content)

    modal = int(data.get("modal_verb_count", 0))
    hedge = int(data.get("hedge_marker_count", 0))
    strong = int(data.get("strong_assertion_count", 0))

    # 派生强度指标（简单、可解释）
    denom = modal + hedge + strong
    strength = (strong - hedge) / denom if denom > 0 else 0.0

    return {
        "modal_verb_count": modal,
        "hedge_marker_count": hedge,
        "strong_assertion_count": strong,
        "epistemic_strength_score": round(strength, 4)
    }

# 测试和使用示例
def main():
    """主测试函数"""
    print("启动优化版特征提取器...")
    print("=" * 70)

    # 创建提取器
    extractor = OptimizedFeatureExtractor(cache_enabled=True)

    # 测试文本（包含之前识别错误的英文反问句）
    test_texts = [
        "难道你不知道这个规定吗？为什么还要这样做？",
        "Isn't it beautiful? What do you think?",
        "这难道不是你的责任吗？你怎么能推卸？",
        "How can you say that? Doesn't it make sense?",
        "今天天气真好，我们去公园吧？",
        "你还没有完成作业吗？这怎么行？",
    ]

    print(f"测试 {len(test_texts)} 个文本...")
    print("-" * 70)

    # 批量提取
    results = extractor.batch_extract(test_texts, batch_size=3)

    # 显示结果
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.summary()}")

        if result.questions:
            print("  详细分析:")
            for j, q in enumerate(result.questions, 1):
                marker = "🔸" if q["is_rhetorical"] else "◦"
                print(f"    {marker} 问题{j}: {q['text']}")
                """
                                if q["is_rhetorical"]:
                    print(f"       理由: {q['reason']} (置信度: {q['confidence']:.2f})")
                """


        print(f"  模型: {result.metadata.get('model')} | "
              f"用时: {result.metadata.get('response_time', 0):.2f}s")

    # 显示统计
    print("\n" + "=" * 70)
    print("使用统计:")
    stats = extractor.get_stats()
    print(f"总请求: {stats['total_requests']}")
    print(f"成功: {stats['successful_requests']}")
    print(f"失败: {stats['failed_requests']}")
    print(f"成功率: {stats['success_rate']:.2%}")
    print(f"缓存命中: {stats['cache_hits']}")
    print(f"总token: {stats['total_tokens']}")

    if stats["models_used"]:
        print("模型使用情况:")
        for model, count in stats["models_used"].items():
            print(f"  {model}: {count}次")


if __name__ == "__main__":
    main()