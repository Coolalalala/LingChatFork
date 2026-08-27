use chrono::Local;
use std::{collections::HashMap, sync::{Arc, OnceLock}};
use tokio::sync::{Mutex, RwLock};
use tokio::task::spawn_blocking;

use engramai::{EmbeddingConfig, Memory, MemoryConfig, MemoryType};

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
            tracing::warn!("(debug) Creating memory for: {}", display_name);
            let config = MemoryConfig {
                embedding: EmbeddingConfig {
                    provider: "ollama".into(),
                    model: "mxbai-embed-large:latest".into(),
                    dimensions: 1024,
                    ..Default::default()
                },
                fts_weight: 0.30,
                embedding_weight: 0.50,
                actr_weight: 0.20,
                ..Default::default()
            };

            use crate::api::data_dir;
            let memory_path = data_dir().join("game_data").join("memories").join(format!("{}.db", sanitize_role_name(&display_name)));

            Memory::new(memory_path.to_str().unwrap_or("debug.json"), Some(config))
                .unwrap_or_else(|e| {
                    eprintln!("Failed to make memory from: {}", e);
                    panic!();
                })
        })
        .await
        .unwrap()
    }
    
    async fn new(display_name: &str) -> Self {
        Self {
            mem: Arc::new(std::sync::Mutex::new(Self::create_memory(display_name).await)),
        }
    }

    /// 写入新片段
    async fn write(&self, summary: String, importance: Option<f64>) -> String {
        tracing::warn!("(debug) writing: {}", summary);
        let mutex = self.mem.clone();
        spawn_blocking(move || {
            let mut mem = mutex.lock().unwrap();
            match mem.add(&summary, MemoryType::Episodic, importance, None, None) {
                Ok(id) => id,
                Err(e) => {
                    tracing::error!("Failed to add to memory: {}", e);
                    "".to_string()
                }
            }
        })
        .await
        .unwrap()
    }

    /// 召回记忆
    async fn recall(&self, max_nodes: usize, query: String, context: Option<Vec<String>>) -> Vec<String> {
        let mutex = self.mem.clone();
        spawn_blocking(move || {
            let mut output: Vec<String> = Vec::with_capacity(max_nodes);
            let mut mem = mutex.lock().unwrap();
            let results = match mem.recall(&query, max_nodes * 2, context, None) {
                Ok(a) => a,
                Err(e) => {
                        tracing::error!("Failed to recall from memory: {}", e);
                    return Vec::new();
                }
            };

            for result in results.iter().take(max_nodes) { 
                output.push(format!(
                    "[confidence = {} ({:.2})] {} [时间：{}]",
                    result.confidence_label, result.confidence, result.record.content, result.record.created_at.with_timezone(&Local).format("%y/%m/%d，%H:%M")
                ));
            }

            output
        })
        .await
        .unwrap()
    }

    /// 沉淀记忆
    async fn consolidate(&self, time: f64) {
        tracing::warn!("(debug) consolidating memory for: {}days", time);
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
}

/// 线程安全包装器，供运行时共享
type SharedMemory = Arc<RwLock<NodeMemory>>;

/// 记忆：按角色分文件存储，键为 display_name
static TREE_MEMS: OnceLock<Mutex<HashMap<String, SharedMemory>>> = OnceLock::new();

fn tree_mems() -> &'static Mutex<HashMap<String, SharedMemory>> {
    TREE_MEMS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 创建共享实例
async fn new_memory(display_name: &str) -> SharedMemory {
    Arc::new(RwLock::new(NodeMemory::new(display_name).await))
}

/// 获取或创建指定角色的记忆实例
async fn get_or_create_mem(display_name: &str) -> SharedMemory {
    let mut map = tree_mems().lock().await;
    if let Some(mem) = map.get(display_name) {
        return mem.clone();
    }
    let mem = new_memory(display_name).await;
    tracing::warn!("(debug) 已创建角色 {} 的记忆", display_name);
    map.insert(display_name.to_string(), mem.clone());
    mem
}

/// 召回角色的节点记忆
pub async fn recall_node_memory(
    display_name: &str,
    max_nodes: usize,
    query: String,
    context: Option<Vec<String>>,
) -> Vec<String> {
    get_or_create_mem(display_name)
        .await
        .read()
        .await
        .recall(max_nodes, query, context)
        .await
}

/// 写入角色的节点记忆
pub async fn write_node_memory(
    display_name: &str,
    summary: String,
    importance: Option<f64>,
) -> String {
    get_or_create_mem(display_name)
        .await
        .write()
        .await
        .write(summary, importance)
        .await
}


/// 沉淀记忆
pub async fn consolidate_memory(display_name: &str, days: f64) {
    get_or_create_mem(display_name)
        .await
        .write()
        .await
        .consolidate(days)
        .await
}