"""MCP 通信验证脚本（D1 门禁）。

验证 MCP stdio server 能否正常启动、工具调用是否正常。

用法：
    python tests/test_mcp_connection.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def test_mcp_connection():
    """测试 MCP stdio 连接与 echo 工具。"""
    print("=" * 60)
    print(" MCP 通信验证")
    print("=" * 60)
    print()

    server_script = Path(__file__).parent.parent / "mcp_server" / "server.py"
    python_exe = sys.executable

    print(f"📡 启动 MCP 服务器: {server_script}")
    print(f"🐍 Python: {python_exe}")
    print()

    try:
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError:
        print("❌ mcp 包未安装，请先运行: pip install mcp")
        return False

    # 服务器参数
    server_params = StdioServerParameters(
        command=python_exe,
        args=[str(server_script)],
        env=None,
    )

    print("🔌 连接中...", end=" ", flush=True)

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            from mcp.client.session import ClientSession

            async with ClientSession(read_stream, write_stream) as session:
                # 初始化
                await session.initialize()
                print("✅ 连接成功")
                print()

                # 列出工具
                print("📋 可用工具:")
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    print(f"   • {tool.name} — {tool.description[:60]}")
                print()

                # 测试 echo 工具
                print("🧪 测试 echo 工具...")
                test_msg = f"Hello LifeOS @ {int(time.time())}"
                t0 = time.time()

                result = await session.call_tool("echo", {"message": test_msg})
                elapsed = int((time.time() - t0) * 1000)

                # 解析返回
                content = result.content
                if content and len(content) > 0:
                    text = content[0].text if hasattr(content[0], "text") else str(content[0])
                    expected = f"ECHO: {test_msg}"

                    if expected in text:
                        print(f"   ✅ echo 正常 ({elapsed}ms)")
                        print(f"   输入: {test_msg}")
                        print(f"   输出: {text}")
                    else:
                        print(f"   ⚠️  返回内容不匹配 ({elapsed}ms)")
                        print(f"   期望包含: {expected}")
                        print(f"   实际: {text[:100]}")
                else:
                    print(f"   ❌ 无返回内容")

                print()

                # 测试 system_now 工具
                print("🧪 测试 system_now 工具...")
                t0 = time.time()
                result2 = await session.call_tool("system_now", {})
                elapsed2 = int((time.time() - t0) * 1000)

                content2 = result2.content
                if content2 and len(content2) > 0:
                    text2 = content2[0].text if hasattr(content2[0], "text") else str(content2[0])
                    print(f"   ✅ system_now 正常 ({elapsed2}ms)")
                    print(f"   输出: {text2[:100]}")
                else:
                    print(f"   ❌ 无返回内容")

                print()

                # 列出资源
                print("📦 可用资源:")
                try:
                    resources_result = await session.list_resources()
                    if resources_result.resources:
                        for res in resources_result.resources:
                            print(f"   • {res.uri} — {res.name}")
                    else:
                        print("   (无)")
                except Exception as e:
                    print(f"   (list_resources 不支持: {e})")

                print()

        print("=" * 60)
        print("✅ MCP 验证通过")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print()
        print("=" * 60)
        print("❌ MCP 验证失败")
        print("=" * 60)
        print()
        print("排查建议：")
        print("1. 确认 mcp 包已安装: pip install mcp")
        print("2. 确认 server.py 路径正确")
        print("3. 在 Windows 上若 stdio 不稳定，可降级为 SSE 模式")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_mcp_connection())
    sys.exit(0 if success else 1)
