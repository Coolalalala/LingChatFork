use chrono::{DateTime, Local, Utc};
use serde_json::Value;
use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, OnceLock},
};
use tokio::sync::{Mutex, RwLock};
use tokio::task::spawn_blocking;

use engramai::{EmbeddingConfig, Memory, MemoryConfig, MemoryRecord, MemoryType};

use crate::ai_service::llm::{
    LlmClient, LlmConfig, LlmSlot, provider_config::resolve_dreams_provider, slot_snapshot,
};

/// 清理角色名中的非法文件名字符，兜底防空/防 `..`。
/// 复用 memory.rs 笔记模块的同名函数逻辑。
fn sanitize_role_name(name: &str) -> String {
    let cleaned: String = name
        .trim()
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-' || *c == ' ')
        .collect();
    let cleaned = cleaned.trim().to_string();
    if cleaned.is_empty() || cleaned == "." || cleaned == ".." {
        "unknown".to_string()
    } else {
        cleaned
    }
}

/// 内存核心结构
pub struct NodeMemory {
    mem: Arc<std::sync::Mutex<Memory>>,
}

impl NodeMemory {
    async fn create_memory(display_name: &str) -> Memory {
        let display_name = display_name.to_string();
        spawn_blocking(move || {
            tracing::info!("Creating memory for: {}", display_name);
            let config = MemoryConfig {
                embedding: EmbeddingConfig {
                    provider: "ollama".into(),
                    model: "mxbai-embed-large:latest".into(),
                    dimensions: 1024,
                    ..Default::default()
                },
                fts_weight: 0.35,
                embedding_weight: 0.45,
                actr_weight: 0.20,
                ..Default::default()
            };

            use crate::api::data_dir;
            let memory_path = data_dir()
                .join("game_data")
                .join("memories")
                .join(format!("{}.db", sanitize_role_name(&display_name)));

            // 确保路径存在
            if let Some(parent) = memory_path.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                        tracing::error!("Failed to create directory: {}", e);
                        panic!();
                    });
                }
            }

            Memory::new(memory_path.to_str().unwrap_or("debug.json"), Some(config)).unwrap_or_else(
                |e| {
                    tracing::error!("Failed to make memory from: {}", e);
                    panic!();
                },
            )
        })
        .await
        .unwrap()
    }

    async fn new(display_name: &str) -> Self {
        Self {
            mem: Arc::new(std::sync::Mutex::new(
                Self::create_memory(display_name).await,
            )),
        }
    }

    /// 写入新片段
    async fn write(
        &self,
        summary: String,
        memory_type: MemoryType,
        importance: Option<f64>,
        namespace: Option<String>,
    ) -> String {
        tracing::info!(
            "(debug) writing: {} | {:?} | {:?}",
            summary,
            importance,
            namespace
        );
        let mutex = self.mem.clone();
        spawn_blocking(move || {
            let mut mem = mutex.lock().unwrap();
            match mem.add_to_namespace(
                &summary,
                memory_type,
                importance,
                None,
                None,
                namespace.as_deref(),
            ) {
                Ok(id) => id,
                Err(e) => {
                    tracing::error!("Failed to add to memory: {}", e);
                    "".to_string()
                },
            }
        })
        .await
        .unwrap()
    }

    /// 召回记忆
    async fn recall(
        &self,
        max_nodes: usize,
        query: String,
        context: Option<Vec<String>>,
    ) -> Vec<String> {
        let mutex = self.mem.clone();
        spawn_blocking(move || {
            let mut output: Vec<String> = Vec::with_capacity(max_nodes);
            let mut mem = mutex.lock().unwrap();
            let results = match mem.recall(&query, max_nodes, context, None) {
                Ok(a) => a,
                Err(e) => {
                    tracing::error!("Failed to recall from memory: {}", e);
                    return Vec::new();
                },
            };

            for result in results {
                output.push(format!(
                    "[confidence = {} ({:.2})] {} [时间：{}]",
                    result.confidence_label,
                    result.confidence,
                    result.record.content,
                    result
                        .record
                        .created_at
                        .with_timezone(&Local)
                        .format("%y/%m/%d，%H:%M")
                ));
            }
            tracing::info!("(debug) recalled: {:?}", output);

            output
        })
        .await
        .unwrap()
    }

    /// 沉淀记忆
    async fn consolidate(&self, time: f64) {
        tracing::info!("(debug) consolidating memory for: {}days", time);
        let mutex = self.mem.clone();
        spawn_blocking(move || {
            let mut mem = mutex.lock().unwrap();
            if let Err(e) = mem.consolidate(time) {
                tracing::error!("Failed to consolidate memory: {}", e);
            }
        })
        .await
        .unwrap()
    }

    /// 获取最近 n 条记忆
    fn recall_recent(&self, n: usize, namespace: Option<&str>) -> Vec<MemoryRecord> {
        let mem = self.mem.lock().unwrap();
        if let Ok(memories) = mem.recall_recent(n, namespace) {
            memories
        } else {
            tracing::error!("Failed to recall recent from memory");
            Vec::new()
        }
    }

    /// 提取距离某个时间最近的 n 条记忆
    fn recall_in_range(
        &self,
        time: DateTime<Utc>,
        n: usize,
        namespace: Option<&str>,
    ) -> Vec<MemoryRecord> {
        let mem = self.mem.lock().unwrap();
        let mut records = match mem.storage().all_in_namespace(namespace) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Failed to load memories: {}", e);
                return Vec::new();
            },
        };

        // Sort by absolute time difference, closest first
        records.sort_by_key(|record| {
            let diff = (record.created_at - time).abs(); // returns TimeDelta
            // `TimeDelta` is `Ord`, so it can be used as a sort key directly
            diff
        });

        records.truncate(n);
        records
    }
}

/// LLM 接口
static DREAMS_CLIENT: LazyLock<LlmSlot> =
    LazyLock::new(|| std::sync::Arc::new(tokio::sync::RwLock::new(None)));

pub async fn set_dreams_client(client: Option<Arc<LlmClient>>) {
    let mut dreams_client = DREAMS_CLIENT.write().await;
    *dreams_client = client;
    tracing::info!("[switch_llm] 发呆 LLM 槽位已热切换");
}

/// 记忆：按角色分文件存储，键为 display_name
static TREE_MEMS: OnceLock<Mutex<HashMap<String, Arc<NodeMemory>>>> = OnceLock::new();

fn tree_mems() -> &'static Mutex<HashMap<String, Arc<NodeMemory>>> {
    TREE_MEMS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 创建共享实例
async fn new_memory(display_name: &str) -> Arc<NodeMemory> {
    Arc::new(NodeMemory::new(display_name).await)
}

/// 获取或创建指定角色的记忆实例
async fn get_or_create_mem(display_name: &str) -> Arc<NodeMemory> {
    let mut map = tree_mems().lock().await;
    if let Some(mem) = map.get(display_name) {
        return mem.clone();
    }
    let mem = new_memory(display_name).await;
    tracing::info!("已创建角色 {} 的记忆", display_name);
    map.insert(display_name.to_string(), mem.clone());
    // 获取最新记忆时间并沉淀记忆
    let recent_memories = mem.recall_recent(1, None);
    if let Some(latest) = recent_memories.first() {
        let duration = Utc::now().signed_duration_since(latest.created_at);
        let days = duration.num_seconds() as f64 / 86400.0; // 转换为天数
        if days > 0.1 {
            mem.consolidate(days).await;
        }
    }
    mem
}

/// 召回角色的节点记忆
pub async fn recall_vector_memory(
    display_name: &str,
    max_nodes: usize,
    query: String,
    context: Option<Vec<String>>,
) -> Vec<String> {
    get_or_create_mem(display_name)
        .await
        .recall(max_nodes, query, context)
        .await
}

/// 写入角色的节点记忆
pub async fn write_vector_memory(
    display_name: &str,
    summary: String,
    importance: Option<f64>,
    namespace: Option<String>,
) -> String {
    write_memory_with_type(
        display_name,
        summary,
        MemoryType::Episodic,
        importance,
        namespace,
    )
    .await
}

/// 写入角色的节点记忆
pub async fn write_memory_with_type(
    display_name: &str,
    summary: String,
    memory_type: MemoryType,
    importance: Option<f64>,
    namespace: Option<String>,
) -> String {
    get_or_create_mem(display_name)
        .await
        .write(summary, memory_type, importance, namespace)
        .await
}

/// 沉淀记忆
pub async fn consolidate_memory(display_name: &str, days: f64) {
    get_or_create_mem(display_name)
        .await
        .consolidate(days)
        .await
}

/// 根据时间获取记忆（默认最近时间）
pub async fn recall_node_memory_in_range(
    display_name: &str,
    time: Option<DateTime<Utc>>,
    max_nodes: usize,
    namespace: Option<&str>,
) -> Vec<MemoryRecord> {
    match time {
        Some(time) => get_or_create_mem(display_name)
            .await
            .recall_in_range(time, max_nodes, namespace),
        None => get_or_create_mem(display_name)
            .await
            .recall_recent(max_nodes, namespace),
    }
}

pub async fn daydream(
    display_name: &str,
    character_prompt: &str,
    reason: &str,
    topic: Option<String>,
    llm: Arc<LlmClient>, // 从 LlmSlot 快照传入
) -> Result<String, anyhow::Error> {
    // 1. 取素材
    if reason.is_empty() {
        return Err(anyhow::anyhow!("reason 不能为空"));
    }
    let mut records = recall_node_memory_in_range(display_name, None, 8, Some("chat_msg")).await;
    records.reverse();
    let mut memories_prompt: String = "以下是你的发呆素材：".to_string();
    for record in records {
        memories_prompt += &format!(
            "\n{}\n[时间：{}]\n---",
            record.content,
            record
                .created_at
                .with_timezone(&Local)
                .format("%y/%m/%d，%H:%M")
        );
    }
    if let Some(query) = topic {
        memories_prompt += &recall_vector_memory(display_name, 8, query, None)
            .await
            .join("\n---\n");
    }

    // 2. 组 task_prompt
    let task_prompt = format!("\n\n你正在自己无监管的思绪中发呆，原因是\"{}\"\n", reason)
        + r#"
接下来无论你产生任何回复都不会被用户观察到，不要对用户说话，不要做出行为或描述场景，只要思考就好。
接下来请自言自语（比如展望未来，复习刚才对话，或者理性分析），发散出大约 3000 字的文案以解决上述情况。
"#;

    let system_prompt = format!("{}{}{}", character_prompt, task_prompt, memories_prompt);

    // 3. 自言自语长文
    let Some(monologue_llm) = slot_snapshot(&DREAMS_CLIENT).await else {
        tracing::error!("发呆 LLM 槽位未配置！");
        return Err(anyhow::anyhow!("发呆 LLM 槽位未配置！"));
    };
    let monologue = monologue_llm
        .complete(&vec![crate::ai_service::types::LlmMessage::system(
            system_prompt,
        )])
        .await?;
    tracing::info!("(debug) 已完成思考：{}", &monologue);

    // 4. 跑 extraction prompt → JSON insights
    let extraction_prompt = format!(
        "{}{}",
        r#"
你是一个专业的【思绪分析专家】。接下来的消息中会包含你自己的思考内容，你的任务是抽取你自己思考内容中的结论。
请从你思考的长文段中抽取 3-10 条信息（如历史经过，个人观点等）
不要记录客观上不可察觉的信息（如思绪的过程）
使用第一人称，主观的语言。

以下是重要度的具体指引参考：
- 重要度 0.1-0.3：不太重要的信息，比如：事情经过，细节，暂时时效性的信息。
- 重要度 0.4-0.6：重要的信息，比如：重要的事情发生，个人观点，爱好，情绪表达等。
- 重要度 0.7-0.9：不能忽视的重要信息，比如：剧烈的情绪，关键的事件，学会的技能等。
越是复杂的信息，重要度越低。例如：
代码的实现 - 复杂，不重要
事件的过程 - 复杂，较不重要
出现的情绪 - 简单，重要


对于每条新信息，请按以下格式输出：
{
    "insights": [
        {
            "content": "<信息内容>",
            "importance": <重要度，0.0-1.0 之间的浮点数>
        },
        ... // 更多信息
    ]
}

"#,
        character_prompt
    );

    let mut attempts = 0;
    let mut response = "".to_string();
    while attempts < 5 {
        match llm
            .complete(&vec![
                crate::ai_service::types::LlmMessage::system(&extraction_prompt),
                crate::ai_service::types::LlmMessage::user(&monologue),
                crate::ai_service::types::LlmMessage::assistant("{"),
            ])
            .await
        {
            Ok(result) => {
                response = result;
                break;
            },
            Err(e) => {
                tracing::warn!("Extraction failed: {}", e);
                attempts += 1;
                continue;
            },
        }
    }
    if response.is_empty() {
        tracing::error!("发呆抽取记忆失败！Aborting...");
        return Ok(monologue);
    }

    if !response.starts_with("{") {
        response = "{".to_string() + &response;
    }

    // 5. Parse insights
    if let Ok(insights) = serde_json::from_str(&response) {
        let insights: Value = insights;
        tracing::info!("Insights: {}", insights);
        for insight in insights["insights"].as_array().unwrap() {
            let (Some(content), Some(importance)) =
                (insight["content"].as_str(), insight["importance"].as_f64())
            else {
                tracing::warn!("跳过格式异常的 insight: {insight}");
                continue; // 跳过坏条目
            };
            let _id = write_memory_with_type(
                display_name,
                content.to_string(),
                MemoryType::Opinion,
                Some(importance),
                Some("insight".to_string()),
            )
            .await;
        }
        consolidate_memory(display_name, 0.2).await;
    } else {
        tracing::error!(
            "Failed to parse insights because of invalid JSON: {}",
            response
        );
    }
    return Ok(monologue);
}
