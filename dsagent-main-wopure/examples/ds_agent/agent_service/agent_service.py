import asyncio
import json
import logging
import time
import os
from datetime import datetime
from typing import AsyncGenerator, Dict, Union
from pathlib import Path

from fastapi import HTTPException

# Lazy import MetaGPT (only when needed)
DSAgentStream = None
LanguageAgentTreeSearchStream = None
AGENT_SERVICE_FILE = None
LLM = None

def _import_metagpt():
    """Lazy import MetaGPT modules to avoid dependency issues."""
    global DSAgentStream, LanguageAgentTreeSearchStream, AGENT_SERVICE_FILE, LLM
    if DSAgentStream is None:
        from metagpt.roles.ds_agent.ds_agent_stream import DSAgentStream as _DSAgentStream
        from metagpt.roles.ds_agent.lats_react_stream import LanguageAgentTreeSearchStream as _LATS
        from metagpt.const import AGENT_SERVICE_FILE as _AGENT_SERVICE_FILE
        from metagpt.llm import LLM as _LLM
        DSAgentStream = _DSAgentStream
        LanguageAgentTreeSearchStream = _LATS
        AGENT_SERVICE_FILE = _AGENT_SERVICE_FILE
        LLM = _LLM

# Try to import Pure AutoGen agents (no MetaGPT fallback)
try:
    from autogen_agent_service_pure import PureAutoGenDSAgent
    AutoGenDSAgent = PureAutoGenDSAgent
    AutoGenLATSAgent = None  # LATS is MetaGPT-specific
    AUTOGEN_ENABLED = True
    logging.info("✅ Pure AutoGen agents loaded (no MetaGPT fallback)")
except ImportError as e:
    AUTOGEN_ENABLED = False
    AutoGenDSAgent = None
    AutoGenLATSAgent = None
    logging.warning(f"⚠️ AutoGen not available: {e}")

# 配置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 框架选择: 'metagpt' 或 'autogen'
AGENT_FRAMEWORK = os.getenv('AGENT_FRAMEWORK', 'metagpt').lower()


# ------------------ SSE格式化工具 ------------------
class StreamFormatter:
    @staticmethod
    def format_data(data: dict) -> str:
        """直接返回JSON数据，不添加SSE格式的头信息"""
        json_data = json.dumps(data, ensure_ascii=False)
        return json_data + "\n"

    @staticmethod
    def format_event(data: dict, event_type: str = "message", event_id: int = None, retry: int = 5000) -> str:
        event = []
        if event_id is not None:
            event.append(f"id: {event_id}")
        if event_type:
            event.append(f"event: {event_type}")
        if retry:
            event.append(f"retry: {retry}")
        json_data = json.dumps(data, ensure_ascii=False)
        event.append(f"data: {json_data}")
        return "\n".join(event) + "\n\n"


# ------------------ 智能体核心类 ------------------
class DSAgent:
    def __init__(self, agent_id: str):
        _import_metagpt()  # Lazy load MetaGPT
        self.agent_id = agent_id
        # 启用RAG和经验功能,传入LLM实例和name
        self.agent = DSAgentStream(
            name=f"DSAgent_{agent_id}",
            llm=LLM(), 
            use_reflection=True, 
            use_rag=True, 
            use_kaggle_exp=True, 
            use_exp_extractor=True
        )
        self._active = False
        self._lock = asyncio.Lock()
        self.event_counter = 0

    async def acquire(self) -> bool:
        async with self._lock:
            if not self._active:
                self._active = True
                return True
            return False

    def release(self):
        self.agent.clear_content()
        self._active = False

    async def process_stream(self, requirement: str) -> AsyncGenerator[dict, None]:
        try:
            async for chunk in self.agent.stream_run(with_message=requirement):
                self.event_counter += 1
                yield {
                    "type": "text_chunk",
                    "content": chunk,
                    "timestamp": datetime.now().isoformat(),
                    "sequence": self.event_counter
                }
                # Use logger instead of print to avoid encoding issues
                logger.debug(f"Chunk {self.event_counter}: {chunk[:100] if len(chunk) > 100 else chunk}")
            yield {
                "type": "system",
                "content": "处理流程已完成",
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Agent {self.agent_id} 处理异常: {str(e)}")
            yield {
                "type": "error",
                "code": 500,
                "content": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            self.release()


class LATSAgent:
    def __init__(self, agent_id: str, use_exp_driven_search=True):
        self.agent_id = agent_id
        
        # Use new dsagent_core LATS implementation
        from dsagent_core.adapters import MetaGPTLATSAdapter
        
        logger.info(f"🌲 Initializing LATS agent {agent_id} with dsagent_core")
        self.agent = MetaGPTLATSAdapter(
            use_exp_driven_search=use_exp_driven_search,
            max_depth=10,
            high_reward_threshold=7.0
        )
        self._active = False
        self._lock = asyncio.Lock()
        self.event_counter = 0

    async def acquire(self) -> bool:
        async with self._lock:
            if not self._active:
                self._active = True
                return True
            return False

    def release(self):
        self.agent.root = None
        self.agent.all_nodes = []
        self.agent.failed_trajectories = []
        self.agent.terminal_nodes = []
        self._active = False

    async def process_stream(self, requirement: str, iterations=10, n_generate_sample=2) -> AsyncGenerator[dict, None]:
        try:
            logger.info(f"🌲 LATS Agent {self.agent_id} starting tree search for: {requirement[:50]}...")
            self.agent.goal = requirement
            
            # Use new enhance_run from dsagent_core adapter
            conclusion = await self.agent.enhance_run(iterations=iterations, n_generate_sample=n_generate_sample)
            
            # Stream the conclusion
            self.event_counter += 1
            yield {
                "type": "text_chunk",
                "content": conclusion,
                "timestamp": datetime.now().isoformat(),
                "sequence": self.event_counter
            }
            
            yield {
                "type": "system",
                "content": "LATS 树搜索已完成",
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Agent {self.agent_id} 处理异常: {str(e)}")
            yield {
                "type": "error",
                "code": 500,
                "content": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            self.release()


# ------------------ API服务封装 ------------------
class AgentServiceProvider:
    def __init__(self, initial_agent_counts: dict = {"ds": 1, "lats": 1}, framework: str = None):
        self.framework = framework or AGENT_FRAMEWORK
        self.agents_pool: Dict[str, Dict[str, Union[DSAgent, LATSAgent]]] = {
            "ds": {},
            "lats": {}
        }
        
        # Log framework selection with both print and logger
        print("=" * 60)
        print(f"🚀 Agent Framework: {self.framework.upper()}")
        print("=" * 60)
        logger.info(f"=" * 60)
        logger.info(f"Agent Framework: {self.framework.upper()}")
        if self.framework == 'autogen' and not AUTOGEN_ENABLED:
            print("⚠️  AutoGen requested but not available, falling back to MetaGPT")
            logger.warning("AutoGen requested but not available, falling back to MetaGPT")
            self.framework = 'metagpt'
        logger.info(f"=" * 60)
        
        self._init_agent_pool(initial_agent_counts)

    def _init_agent_pool(self, initial_agent_counts: dict):
        for mode, count in initial_agent_counts.items():
            for _ in range(count):
                agent_id = f"{mode.upper()}-{len(self.agents_pool[mode]) + 1}-{int(time.time())}"
                
                if self.framework == 'autogen' and AUTOGEN_ENABLED:
                    # Use AutoGen agents
                    if mode == "ds":
                        print(f"🤖 Creating AutoGen DS agent {agent_id}...")
                        text_exp_path = Path("examples/data/exp_bank/plan_exp.json")
                        workflow_exp_path = Path("examples/data/exp_bank/workflow_exp2_clean_new.json")
                        self.agents_pool[mode][agent_id] = AutoGenDSAgent(
                            agent_id,
                            text_exp_path=text_exp_path if text_exp_path.exists() else None,
                            workflow_exp_path=workflow_exp_path if workflow_exp_path.exists() else None
                        )
                        print(f"✅ Initialized AutoGen DS agent {agent_id}")
                        logger.info(f"✅ Initialized AutoGen {mode} agent {agent_id}")
                    elif mode == "lats":
                        # ✨ NEW: AutoGen now supports LATS via dsagent_core!
                        from dsagent_core.adapters import create_autogen_lats
                        import os
                        
                        api_key = os.getenv('DASHSCOPE_API_KEY')
                        if not api_key:
                            print("⚠️  DASHSCOPE_API_KEY not set, cannot create AutoGen LATS agent")
                            logger.warning("⚠️  DASHSCOPE_API_KEY not set, cannot create AutoGen LATS agent")
                            continue
                        
                        print(f"🌲 Creating AutoGen LATS agent {agent_id} with dsagent_core...")
                        logger.info(f"🌲 Creating AutoGen LATS agent {agent_id} with dsagent_core")
                        # Create a wrapper class for AutoGen LATS
                        class AutoGenLATSAgentWrapper:
                            def __init__(self, agent_id, lats_adapter):
                                self.agent_id = agent_id
                                self.lats_adapter = lats_adapter
                                self._active = False
                                self._lock = asyncio.Lock()
                                self.event_counter = 0
                            
                            async def acquire(self):
                                async with self._lock:
                                    if not self._active:
                                        self._active = True
                                        return True
                                    return False
                            
                            def release(self):
                                self._active = False
                            
                            async def process_stream(self, requirement: str, iterations=10, n_generate_sample=2):
                                try:
                                    print(f"🌲 AutoGen LATS Agent {self.agent_id} starting tree search")
                                    logger.info(f"🌲 AutoGen LATS Agent {self.agent_id} starting tree search")
                                    
                                    # Add timeout protection
                                    import asyncio
                                    result = await asyncio.wait_for(
                                        self.lats_adapter.run_and_format(
                                            goal=requirement,
                                            iterations=iterations,
                                            n_generate_sample=n_generate_sample
                                        ),
                                        timeout=300.0  # 5 minute timeout
                                    )
                                    
                                    # Format result
                                    output = f"LATS 树搜索结果:\n"
                                    output += f"探索节点数: {result['nodes_explored']}\n"
                                    output += f"最佳奖励: {result['best_reward']:.2f}/10\n"
                                    output += f"解决方案深度: {result['depth']}\n\n"
                                    output += f"解决方案步骤:\n"
                                    for i, step in enumerate(result['solution_steps'], 1):
                                        thought = step.get('thought', {})
                                        if isinstance(thought, dict):
                                            thought_text = thought.get('thought', str(thought))
                                        else:
                                            thought_text = str(thought)
                                        output += f"{i}. {thought_text[:200]}\n"
                                    output += f"\n最终输出:\n{result['final_output'][:1000]}"
                                    
                                    self.event_counter += 1
                                    yield {
                                        "type": "text_chunk",
                                        "content": output,
                                        "timestamp": datetime.now().isoformat(),
                                        "sequence": self.event_counter
                                    }
                                    
                                    yield {
                                        "type": "system",
                                        "content": "AutoGen LATS 树搜索已完成",
                                        "status": "completed",
                                        "timestamp": datetime.now().isoformat()
                                    }
                                except asyncio.TimeoutError:
                                    error_msg = "LATS 搜索超时 (5分钟)"
                                    print(f"⚠️  {error_msg}")
                                    logger.error(f"AutoGen LATS Agent {self.agent_id} timeout")
                                    yield {
                                        "type": "error",
                                        "code": 504,
                                        "content": error_msg,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                except Exception as e:
                                    logger.error(f"AutoGen LATS Agent {self.agent_id} 处理异常: {str(e)}")
                                    yield {
                                        "type": "error",
                                        "code": 500,
                                        "content": str(e),
                                        "timestamp": datetime.now().isoformat()
                                    }
                                finally:
                                    self.release()
                        
                        lats_adapter = create_autogen_lats(api_key=api_key, model="qwen-plus")
                        self.agents_pool[mode][agent_id] = AutoGenLATSAgentWrapper(agent_id, lats_adapter)
                        print(f"✅ Initialized AutoGen LATS agent {agent_id}")
                        logger.info(f"✅ Initialized AutoGen LATS agent {agent_id}")
                else:
                    # Use MetaGPT agents (default)
                    if mode == "ds":
                        self.agents_pool[mode][agent_id] = DSAgent(agent_id)
                        print(f"✅ Initialized MetaGPT DS agent {agent_id}")
                    elif mode == "lats":
                        print(f"🌲 Creating MetaGPT LATS agent {agent_id}...")
                        self.agents_pool[mode][agent_id] = LATSAgent(agent_id)
                        print(f"✅ Initialized MetaGPT LATS agent {agent_id}")
                    logger.info(f"✅ Initialized MetaGPT {mode} agent {agent_id}")

    async def get_idle_agent(self, mode: str = "ds") -> Union[DSAgent, LATSAgent]:
        if mode not in self.agents_pool:
            raise HTTPException(400, detail=f"Invalid agent mode: {mode}")

        start_time = time.time()
        while time.time() - start_time < 5:
            for agent in self.agents_pool[mode].values():
                if await agent.acquire():
                    return agent
            await asyncio.sleep(0.1)
        raise HTTPException(503, detail=f"No available {mode} agents")

    async def stream_generator(self, agent: Union[DSAgent, LATSAgent], query: str, session_id) -> AsyncGenerator[str, None]:
        try:
            # Ensure MetaGPT is imported for AGENT_SERVICE_FILE
            _import_metagpt()
            
            yield StreamFormatter.format_data({
                "status": "start",
                "query": query,
                "agent_id": agent.agent_id,
                "timestamp": datetime.now().isoformat()
            })

            # 请求重构 - 获取上传的文件信息
            files = []
            file_names = []
            
            # Use AGENT_SERVICE_FILE if available, otherwise use default path
            if AGENT_SERVICE_FILE is not None:
                session_dir = AGENT_SERVICE_FILE / session_id
            else:
                # Fallback to default path
                session_dir = Path("storage") / session_id
            
            logger.info(f"Checking session directory: {session_dir}, exists={session_dir.exists()}, is_dir={session_dir.is_dir() if session_dir.exists() else 'N/A'}")
            
            # 如果指定的 session 目录不存在，尝试查找最新的 session 目录
            if not (session_dir.exists() and session_dir.is_dir()):
                logger.warning(f"Session directory '{session_id}' does not exist, looking for latest session...")
                parent_dir = AGENT_SERVICE_FILE if AGENT_SERVICE_FILE is not None else Path("storage")
                if parent_dir.exists():
                    session_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]
                    if session_dirs:
                        # 按修改时间排序，取最新的
                        session_dir = max(session_dirs, key=lambda d: d.stat().st_mtime)
                        logger.info(f"Using latest session directory: {session_dir}")
            
            if session_dir.exists() and session_dir.is_dir():
                files = [file for file in session_dir.iterdir() if file.is_file()]
                file_names = [file.name for file in files]
                logger.info(f"Found {len(files)} files: {file_names}")
            else:
                logger.warning(f"No valid session directory found")
            
            # 如果有上传的文件,修改查询以包含文件绝对路径
            if files:
                # 使用绝对路径,这样Jupyter kernel可以直接访问
                file_paths = [str(file.absolute()) for file in files]
                file_names = [file.name for file in files]
                
                # 告诉Agent使用绝对路径加载文件
                prompt = "You are required to {question}. The data files available are: {file_list}. Please load and analyze these files using their full paths."
                file_list_str = ', '.join(f"'{name}' at '{path}'" for name, path in zip(file_names, file_paths))
                query = prompt.format(
                    question=query, 
                    file_list=file_list_str
                )
                logger.info(f"Modified query with file paths: {query[:200]}...")
            else:
                # 没有文件时的默认提示
                query = f"You are required to {query}. Note: No data files were uploaded. [no files]"
                logger.info(f"No files found, using default query: {query[:200]}...")

            if isinstance(agent, DSAgent):
                async for chunk in agent.process_stream(query):
                    yield StreamFormatter.format_data(chunk)
            elif isinstance(agent, LATSAgent):
                async for chunk in agent.process_stream(query):
                    yield StreamFormatter.format_data(chunk)
            elif AUTOGEN_ENABLED and isinstance(agent, (AutoGenDSAgent, AutoGenLATSAgent)):
                # Handle AutoGen agents
                if isinstance(agent, AutoGenDSAgent):
                    async for chunk in agent.process_stream(query):
                        yield StreamFormatter.format_data(chunk)
                else:  # AutoGenLATSAgent
                    async for chunk in agent.process_stream(query):
                        yield StreamFormatter.format_data(chunk)

            yield StreamFormatter.format_data({
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield StreamFormatter.format_data({
                "error_type": "stream_error",
                "code": 500,
                "message": "流式处理中断",
                "detail": str(e)
            })

# ------------------ FastAPI Routes ------------------
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI(title="DS Agent Service")

# Agent service instance
service = AgentServiceProvider()

@app.post("/query")
async def query_stream(request: Request):
    """Stream processing endpoint"""
    data = await request.json()
    query_text = data.get("query", "")
    session_id = data.get("session_id", "default-session")
    mode = data.get("mode", "ds")  # 'ds' or 'lats'
    
    logger.info(f"📥 Received query: {query_text[:100]}...")
    
    # Get an available agent from the pool
    agent = await service.get_idle_agent(mode)
    
    return StreamingResponse(
        service.stream_generator(agent, query_text, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/v1/dsagent/stream")
async def dsagent_stream(request: Request):
    """DSAgent stream endpoint (frontend compatible)"""
    data = await request.json()
    query_text = data.get("query", "")
    session_id = data.get("session_id", "default-session")
    mode = data.get("mode", "ds")  # 'ds' or 'lats'
    
    logger.info(f"📥 DSAgent stream request: {query_text[:100]}...")
    
    # Get an available agent from the pool
    agent = await service.get_idle_agent(mode)
    
    return StreamingResponse(
        service.stream_generator(agent, query_text, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/stop/{agent_id}")
async def stop_agent(agent_id: str):
    """Stop an agent"""
    if service.stop_agent(agent_id):
        return {"status": "success", "message": f"Agent {agent_id} stopped"}
    return {"status": "error", "message": f"Agent {agent_id} not found"}

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "framework": os.getenv("AGENT_FRAMEWORK", "metagpt"),
        "autogen_enabled": AUTOGEN_ENABLED
    }

if __name__ == "__main__":
    logger.info(f"🚀 Starting Agent Service...")
    logger.info(f"📋 Agent Framework: {os.getenv('AGENT_FRAMEWORK', 'METAGPT').upper()}")
    logger.info(f"🤖 AutoGen Enabled: {AUTOGEN_ENABLED}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
