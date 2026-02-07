"""
Provides OCaml specific instantiation of the LanguageServer class using ocamllsp.
Contains various configurations and settings specific to OCaml (.ml, .mli files).
"""

import logging
import os
import pathlib
import shutil
import subprocess
import threading
from typing import Any

from solidlsp.ls import SolidLanguageServer
from solidlsp.ls_config import LanguageServerConfig
from solidlsp.ls_exceptions import SolidLSPException
from solidlsp.lsp_protocol_handler.lsp_types import InitializeParams
from solidlsp.lsp_protocol_handler.server import ProcessLaunchInfo
from solidlsp.settings import SolidLSPSettings

log = logging.getLogger(__name__)


class OCamlLanguageServer(SolidLanguageServer):
    """
    Provides OCaml specific instantiation of the LanguageServer class using ocamllsp.
    Supports .ml and .mli files with features like code completion, go-to-definition,
    find references, hover, diagnostics, and document symbols.

    Requires ocaml-lsp-server to be installed (via opam install ocaml-lsp-server).
    """

    def __init__(self, config: LanguageServerConfig, repository_root_path: str, solidlsp_settings: SolidLSPSettings):
        """
        Creates an OCamlLanguageServer instance. This class is not meant to be instantiated directly.
        Use LanguageServer.create() instead.
        """
        ocamllsp_executable_path = self._setup_runtime_dependencies(config, solidlsp_settings)
        super().__init__(
            config,
            repository_root_path,
            ProcessLaunchInfo(cmd=ocamllsp_executable_path, cwd=repository_root_path),
            "ocaml",
            solidlsp_settings,
        )
        self.server_ready = threading.Event()
        self.initialize_searcher_command_available = threading.Event()

    def is_ignored_dirname(self, dirname: str) -> bool:
        return super().is_ignored_dirname(dirname) or dirname in [
            "_build",
            "_opam",
            ".merlin",
        ]

    @classmethod
    def _setup_runtime_dependencies(cls, config: LanguageServerConfig, solidlsp_settings: SolidLSPSettings) -> str:
        """
        Setup runtime dependencies for OCaml Language Server and return the command to start the server.
        """
        # ocamllsp is typically installed via opam and available in the opam switch PATH.
        # We first check the current PATH, then try to locate it via opam.
        ocamllsp_path = shutil.which("ocamllsp")

        if not ocamllsp_path:
            # Try to find it in the default opam switch
            try:
                result = subprocess.run(
                    ["opam", "var", "bin"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                opam_bin = result.stdout.strip()
                candidate = os.path.join(opam_bin, "ocamllsp")
                if os.path.exists(candidate):
                    ocamllsp_path = candidate
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        if not ocamllsp_path:
            # Fallback: check well-known opam paths directly
            home = os.path.expanduser("~")
            for switch_name in ["default", "."]:
                candidate = os.path.join(home, ".opam", switch_name, "bin", "ocamllsp")
                if os.path.exists(candidate):
                    ocamllsp_path = candidate
                    break

        if not ocamllsp_path:
            raise FileNotFoundError(
                "ocamllsp is not installed or not in PATH.\n"
                "Please install ocaml-lsp-server:\n"
                "  opam install ocaml-lsp-server\n"
                "Then ensure the opam environment is loaded:\n"
                "  eval $(opam env)\n"
                "See https://github.com/ocaml/ocaml-lsp for more details."
            )

        log.info(f"Using ocamllsp at {ocamllsp_path}")
        return ocamllsp_path

    @staticmethod
    def _get_initialize_params(repository_absolute_path: str) -> InitializeParams:
        """
        Returns the initialize params for the OCaml Language Server.
        """
        root_uri = pathlib.Path(repository_absolute_path).as_uri()
        initialize_params = {
            "processId": os.getpid(),
            "rootPath": repository_absolute_path,
            "rootUri": root_uri,
            "workspaceFolders": [
                {
                    "uri": root_uri,
                    "name": os.path.basename(repository_absolute_path),
                }
            ],
            "capabilities": {
                "textDocument": {
                    "synchronization": {
                        "dynamicRegistration": True,
                        "willSave": True,
                        "willSaveWaitUntil": True,
                        "didSave": True,
                    },
                    "completion": {
                        "dynamicRegistration": True,
                        "completionItem": {
                            "snippetSupport": True,
                            "commitCharactersSupport": True,
                            "documentationFormat": ["markdown", "plaintext"],
                            "deprecatedSupport": True,
                        },
                    },
                    "hover": {
                        "dynamicRegistration": True,
                        "contentFormat": ["markdown", "plaintext"],
                    },
                    "signatureHelp": {
                        "dynamicRegistration": True,
                        "signatureInformation": {
                            "documentationFormat": ["markdown", "plaintext"],
                        },
                    },
                    "definition": {"dynamicRegistration": True},
                    "typeDefinition": {"dynamicRegistration": True},
                    "implementation": {"dynamicRegistration": True},
                    "references": {"dynamicRegistration": True},
                    "documentHighlight": {"dynamicRegistration": True},
                    "documentSymbol": {
                        "dynamicRegistration": True,
                        "hierarchicalDocumentSymbolSupport": True,
                        "symbolKind": {"valueSet": list(range(1, 27))},
                    },
                    "codeAction": {
                        "dynamicRegistration": True,
                        "codeActionLiteralSupport": {
                            "codeActionKind": {
                                "valueSet": [
                                    "",
                                    "quickfix",
                                    "refactor",
                                    "refactor.extract",
                                    "refactor.inline",
                                    "refactor.rewrite",
                                    "source",
                                ]
                            }
                        },
                    },
                    "codeLens": {"dynamicRegistration": True},
                    "formatting": {"dynamicRegistration": True},
                    "rename": {"dynamicRegistration": True, "prepareSupport": True},
                    "publishDiagnostics": {
                        "relatedInformation": True,
                        "tagSupport": {"valueSet": [1, 2]},
                    },
                    "foldingRange": {
                        "dynamicRegistration": True,
                        "lineFoldingOnly": True,
                    },
                    "selectionRange": {"dynamicRegistration": True},
                },
                "workspace": {
                    "applyEdit": True,
                    "workspaceEdit": {"documentChanges": True},
                    "didChangeConfiguration": {"dynamicRegistration": True},
                    "didChangeWatchedFiles": {"dynamicRegistration": True},
                    "symbol": {"dynamicRegistration": True},
                    "executeCommand": {"dynamicRegistration": True},
                    "configuration": True,
                    "workspaceFolders": True,
                },
                "window": {
                    "workDoneProgress": True,
                },
            },
        }
        return initialize_params  # type: ignore

    def _start_server(self) -> None:
        """
        Starts the OCaml Language Server, waits for the server to be ready.
        """

        def handle_register_capability(params: dict) -> None:
            assert "registrations" in params
            for registration in params["registrations"]:
                if registration["method"] == "workspace/executeCommand":
                    self.initialize_searcher_command_available.set()
            return

        def handle_workspace_configuration(params: dict) -> list:
            items = params.get("items", [])
            return [None] * len(items)

        def do_nothing(params: Any) -> None:
            return

        def window_log_message(msg: dict) -> None:
            log.info(f"ocamllsp: window/logMessage: {msg}")

        self.server.on_request("client/registerCapability", handle_register_capability)
        self.server.on_request("client/unregisterCapability", do_nothing)
        self.server.on_request("workspace/configuration", handle_workspace_configuration)
        self.server.on_request("window/workDoneProgress/create", do_nothing)
        self.server.on_notification("window/logMessage", window_log_message)
        self.server.on_notification("window/showMessage", window_log_message)
        self.server.on_notification("$/progress", do_nothing)
        self.server.on_notification("textDocument/publishDiagnostics", do_nothing)

        log.info("Starting ocamllsp server process")

        try:
            self.server.start()
        except Exception as e:
            log.error(f"Failed to start ocamllsp process: {e}")
            raise SolidLSPException(f"Failed to start ocamllsp: {e}")

        initialize_params = self._get_initialize_params(self.repository_root_path)

        log.info("Sending initialize request to ocamllsp")
        try:
            init_response = self.server.send.initialize(initialize_params)
            log.debug(f"Received initialize response from ocamllsp: {init_response}")
        except Exception as e:
            raise SolidLSPException(
                f"Failed to initialize ocamllsp for {self.repository_root_path}: {e}"
            ) from e

        assert "capabilities" in init_response

        self.server.notify.initialized({})

        self.server_ready.set()
        log.info("ocamllsp initialization complete")
