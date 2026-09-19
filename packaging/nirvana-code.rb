# Homebrew formula. Host it in a tap (e.g. github.com/niravlekinwala/homebrew-nirvana)
# as Formula/nirvana-code.rb, then: brew install niravlekinwala/nirvana/nirvana-code
#
# Fill in `url` and `sha256` from a GitHub release produced by
# scripts/build-release.sh (its .sha256 file has the value). Building from
# source is also supported via `--HEAD`.
class NirvanaCode < Formula
  desc "Local coding assistant for Apple Silicon: TUI, web UI and OpenAI-compatible API on llama.cpp/Metal"
  homepage "https://github.com/niravlekinwala/nirvana-code"
  version "0.3.0"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/niravlekinwala/nirvana-code/releases/download/v#{version}/nirvana-code-#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "7e6a30293795bec77e0c91ed342da27cd8176c612bfca07aa9f84e1c57a3f2c3"
    end
  end

  head do
    url "https://github.com/niravlekinwala/nirvana-code.git", branch: "main"
    depends_on "rust" => :build
    depends_on "cmake" => :build
  end

  depends_on arch: :arm64
  depends_on :macos => :sonoma

  def install
    if build.head?
      # Portable baseline so the build runs on every M-series chip
      ENV["RUSTFLAGS"] = ""
      ENV["GGML_CPU_ARM_ARCH"] = "armv8.4-a+dotprod+fp16"
      system "cargo", "install", "--locked", "--root", prefix, "--path", "."
    else
      if File.exist?("nirvana-code-#{version}-aarch64-apple-darwin")
        bin.install "nirvana-code-#{version}-aarch64-apple-darwin" => "nirvana-code"
      else
        bin.install "nirvana-code"
      end
    end
  end

  def caveats
    <<~EOS
      Download a model to get started:
        nirvana-code download qwen-coder-3b
      Then:
        nirvana-code run        # terminal UI
        nirvana-code web        # browser UI
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/nirvana-code --version")
  end
end
