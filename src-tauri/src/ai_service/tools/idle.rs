use async_trait::async_trait;
use std::time::Duration;
use serde_json::{json, Value};

use tauri::Manager;

use crate::ai_service::types::ToolDefinition;
use crate::ai_service::game_system::vector_memory::daydream;
use crate::ai_service::llm::slot_snapshot;
use crate::AppState;

use super::game_status_handle;
use super::executor::{Tool, ToolContext, ToolError, ToolResult};




/// 校验参数为 JSON object 并返回引用。
fn require_object<'a>(
    arguments: &'a Value,
    tool: &str,
) -> Result<&'a serde_json::Map<String, Value>, ToolError> {
    arguments
        .as_object()
        .ok_or_else(|| ToolError::InvalidArguments(format!("{tool} 参数必须是 JSON object")))
}




/// idle: 主动发呆
pub struct Idle;

#[async_trait]
impl Tool for Idle { 
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "idle",
            "开始发呆。当你不想说话，或者想深度思考/回忆时，请使用此工具。",
            json!({
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "发呆原因（如无聊，没事做），随便回答即可。可以加入发呆的具体内容，比如思考某个问题（如：想思考如何解决哥德巴赫猜想）。"},
                    "topic": {"type": "string", "description": "想要思考的主题（可选），发呆时会回想起相关记忆"},
                },
                "required": ["reason"],
                "additionalProperties": false
            }),
        )
    }

    fn timeout_hint(&self) -> Option<Duration> {
        Some(Duration::from_secs(180)) // 发呆可能需要较长时间
    }

    async fn execute(
        &self,
        context: &ToolContext,
        arguments: Value,
    ) -> Result<ToolResult, ToolError> {
        let obj = require_object(&arguments, "idle")?;
        let reason = obj
            .get("reason")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| ToolError::InvalidArguments("idle 需要 reason".into()))?;
        let topic = obj
            .get("topic")
            .and_then(Value::as_str)
            .map(str::to_string);

        tracing::info!("idle: {} | {:?}", reason, topic);
        // 宿主状态
        let app = context.require_app()?;
        let state = app.state::<AppState>();
        let db = state.db.clone();

        let gs = game_status_handle(&app).await;
        let display_name = {
            let mut gs = gs.lock().await;
            let Some(role_id) = gs.current_role_id else {
                return Err(ToolError::Execution("当前没有对话角色".into()));
            };
            gs.get_role(&db, role_id)
                .await
                .map_err(|e| ToolError::Execution(format!("获取当前角色失败: {e}")))?
                .display_name
                .clone()
                .ok_or_else(|| ToolError::Execution("当前角色没有展示名".into()))?
        };
        let character_prompt = {
            let ai = state.data().ai_service.lock().await;
            ai.settings
                .as_ref()
                .and_then(|s| s.system_prompt.clone())
                .ok_or_else(|| ToolError::Execution("当前角色没有 system_prompt".into()))?
        };

        // 发呆
        let result = daydream(
            &display_name,
            &character_prompt, 
            &reason, 
            topic, 
            slot_snapshot(&state.data().chat.llm)
            .await
            .ok_or_else(|| ToolError::Execution("LLM 未配置".into()))?
        ).await.map_err(|e| ToolError::Execution(format!("发呆失败: {e}")))?;

        Ok(json!({"info":"发呆已完成。用户不应看见你的内心独白 \"monologue\"，所以你不必汇报此次工具结果。", "monologue": result}))
    }
}