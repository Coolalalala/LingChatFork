use chrono::Local;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
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
    mem: Arc<Mutex<Memory>>,
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
    
    pub async fn new(display_name: &str) -> Self {
        Self {
            mem: Arc::new(Mutex::new(Self::create_memory(display_name).await)),
        }
    }

    /// 写入新对话片段：生成摘要并挂载到内存
    pub async fn write(&self, summary: String, importance: Option<f64>) -> String {
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
    pub async fn recall(&self, max_nodes: usize, query: String, context: Option<Vec<String>>) -> Vec<String> {
        tracing::warn!("(debug) recalling: {}", query);
        let mutex = self.mem.clone();
        spawn_blocking(move || {
            let mut output: Vec<String> = Vec::with_capacity(max_nodes);

            let mut mem = mutex.lock().unwrap();
            if let Err(e) = mem.consolidate(0.01) {
                tracing::error!("Failed to consolidate memory: {}", e);
            }
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
}

/// 线程安全包装器，供运行时共享
pub type SharedMemory = Arc<RwLock<NodeMemory>>;

/// 创建共享实例
pub async fn new_memory(display_name: &str) -> SharedMemory {
    Arc::new(RwLock::new(NodeMemory::new(display_name).await))
}