// Native attachment extraction helper. Compiled once by build.rs and embedded
// in the nirvana-code binary; replaces the previous per-call `swift -e` JIT.
//
//   nirvana-extract pdf <file>   → "PAGES:<n>\n---CONTENT---\n<text>"
//   nirvana-extract ocr <file>   → recognised lines, one per line

import Foundation
import PDFKit
import Vision
import AppKit

let args = CommandLine.arguments
guard args.count == 3 else {
    fputs("usage: nirvana-extract <pdf|ocr> <file>\n", stderr)
    exit(2)
}
let url = URL(fileURLWithPath: args[2])

switch args[1] {
case "pdf":
    guard let doc = PDFDocument(url: url) else {
        fputs("ERR: failed to open PDF document\n", stderr)
        exit(1)
    }
    let count = doc.pageCount
    print("PAGES:\(count)")
    var fullText = ""
    let maxPages = min(count, 50)
    for i in 0..<maxPages {
        if let page = doc.page(at: i), let text = page.string {
            fullText += "\n[Page \(i + 1)]\n"
            fullText += text
            if fullText.count > 25000 {
                fullText += "\n[... Document truncated at 25,000 characters to fit model context ...]\n"
                break
            }
        }
    }
    print("---CONTENT---")
    print(fullText)

case "ocr":
    guard let img = NSImage(contentsOf: url),
          let tiff = img.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiff),
          let cgImage = bitmap.cgImage else {
        fputs("ERR: failed to decode image\n", stderr)
        exit(1)
    }
    let request = VNRecognizeTextRequest { (req, _) in
        guard let observations = req.results as? [VNRecognizedTextObservation] else { return }
        for obs in observations {
            if let top = obs.topCandidates(1).first {
                print(top.string)
            }
        }
    }
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    try? handler.perform([request])

default:
    fputs("unknown mode \(args[1])\n", stderr)
    exit(2)
}
