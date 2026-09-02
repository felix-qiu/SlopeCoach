import SwiftUI

struct StatusBadge: View {
    let status: AnalysisStatus

    var body: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(tint)
                .frame(width: 5, height: 5)

            Text(title)
                .font(.system(size: 9, weight: .bold))
                .tracking(0.35)
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(tint.opacity(0.11), in: Capsule())
    }

    private var title: String {
        switch status {
        case .ready:
            "READY"
        case .partial:
            "PARTIAL"
        }
    }

    private var tint: Color {
        switch status {
        case .ready:
            .slopeSuccess
        case .partial:
            .slopeWarning
        }
    }
}
